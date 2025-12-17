import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from util.TripleDataset import TripleDataset
from Model.model import GCN_Transformer
from layers.module import MixPool, DualStreamAdaptiveFusion 
#---------------------------------------------
# 辅助函数
#---------------------------------------------
def print_model_structure(model):
    print("=" * 40)
    print(f"模型结构: {type(model).__name__}")
    print("=" * 40)
    print(model)
    print("-" * 40)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"网络层数: {len(list(model.modules())) - 1}")
    print("=" * 40)

def check_gpu_status():
    """检查GPU状态并打印详细信息"""
    print("=" * 50)
    print("GPU 状态检查")
    print("=" * 50)
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")
    
    if cuda_available:
        # GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"可用GPU数量: {gpu_count}")
        
        # 当前GPU
        current_device = torch.cuda.current_device()
        print(f"当前GPU设备: {current_device}")
        
        # GPU信息
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
            
            # GPU内存信息
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            
            print(f"  已分配内存: {memory_allocated:.2f} GB")
            print(f"  缓存内存: {memory_cached:.2f} GB")  
            print(f"  总内存: {memory_total:.2f} GB")
    else:
        print("未检测到CUDA设备，将使用CPU")
        
    # 检查cuDNN
    cudnn_enabled = torch.backends.cudnn.enabled
    print(f"cuDNN 是否启用: {cudnn_enabled}")
    if cudnn_enabled:
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    
    print("=" * 50)
    return cuda_available

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='多模态图卷积网络训练')
    parser.add_argument('--seed', type=int, default=369, help='random seed')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # 从0.0025降低到0.001
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')  # 从0.0001增加到0.001
    parser.add_argument('--epochs', type=int, default=300, help='max training epochs')  # 从200增加到300
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--patience', type=int, default=40, help='patience for early stopping')  # 从50减少到40
    parser.add_argument('--input_dim', type=int, default=170, help='输入特征维度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')  # 从64增加到128
    parser.add_argument('--num_classes', type=int, default=2, help='分类类别数')
    parser.add_argument('--output_dir', type=str, default='test_results', help='输出目录')
    parser.add_argument('--target_acc', type=float, default=0.90, help='目标准确率')
    parser.add_argument('--max_attempts', type=int, default=1, help='最大尝试次数')
    parser.add_argument('--stop_on_target', action='store_true', help='达到目标准确率时停止')
    parser.add_argument('--test_loop', action='store_true', help='测试循环功能（不实际训练）')
    parser.add_argument('--no_cross_attention', action='store_true',
                        help='使用无交叉注意力的消融模型(MGTAF_without_BiAttention)')
    return parser.parse_args()

#---------------------------------------------
# 评估和训练函数
#---------------------------------------------
def evaluate(model, loader, criterion, device, return_probs=False):
    model.eval()
    correct = 0
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if isinstance(out, tuple):
                logits, _ = out  
            else:
                logits = out
                
            loss = criterion(logits, data.y.long())
            total_loss += loss.item()
            pred = logits.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            
            # 获取概率用于ROC曲线
            probs = torch.softmax(logits, dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader)
    
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # 计算AUC
    try:
        auc_score = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
    except:
        auc_score = 0.0
    
    if return_probs:
        return acc, avg_loss, precision, recall, f1, auc_score, all_labels, all_preds, all_probs
    else:
        return acc, avg_loss, precision, recall, f1

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        output = model(batch)
        # 确保标签是 long 类型
        batch.y = batch.y.long()
        loss = criterion(output, batch.y)
        
        # 添加L2正则化损失
        if hasattr(model, 'reg_loss'):
            loss += model.reg_loss
        
        loss.backward()
        
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(batch.y).sum().item()
        total += batch.y.size(0)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    # 计算其他指标
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

#---------------------------------------------
# 主训练循环
#---------------------------------------------
def train_with_seed(args, seed, run_id):
    set_seed(seed)
    
    alpha_history = {
        'mixpool_pc': [],
        'mixpool_sr': [],
        'mixpool_gcm': [],
    }
    
    output_dir = Path(args.output_dir) / f"run_{run_id:03d}_seed_{seed}"
    output_dir.mkdir(exist_ok=True, parents=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # 准备设备
    print(f"\n设备配置:")
    print(f"请求使用设备: {args.device}")
    
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU替代")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"实际使用设备: {device}")
    
    # 如果使用GPU，打印更多信息
    if device.type == 'cuda':
        print(f"GPU名称: {torch.cuda.get_device_name(device)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    
    print("-" * 30)
    
    split_info = {}
    
    try:
        dataset = TripleDataset(rf"data", 'PC', 'SR', 'GCM', 0.1, 0.1, 0.2, k=0, augment=True)  # 启用数据增强
        
        num_training = int(len(dataset) * 0.7)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - (num_training + num_val)

        generator = torch.Generator().manual_seed(seed)
        training_set, validation_set, test_set = random_split(
            dataset, [num_training, num_val, num_test], generator=generator)

        split_info["train_indices"] = training_set.indices
        split_info["val_indices"] = validation_set.indices
        split_info["test_indices"] = test_set.indices
        
        with open(output_dir / "data_split.json", "w") as f:
            json.dump(split_info, f)
            
        train_loader = DataLoader(training_set, batch_size=args.batch_size, 
                                 shuffle=True, follow_batch=['x1', 'x2', 'x3'])
        val_loader = DataLoader(validation_set, batch_size=args.batch_size, 
                               shuffle=False, follow_batch=['x1', 'x2', 'x3'])
        test_loader = DataLoader(test_set, batch_size=args.batch_size, 
                                shuffle=False, follow_batch=['x1', 'x2', 'x3'])
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return 0, 0, seed, {}
    
    model = GCN_Transformer(args).to(device)

    print_model_structure(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=15, verbose=True, min_lr=1e-6)  # 调整调度器参数
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    

    train_metrics = {
        'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []
    }
    val_metrics = {
        'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []
    }
    

    print(f"开始训练, 总轮次: {args.epochs}")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device)
        
        val_acc, val_loss, val_precision, val_recall, val_f1 = evaluate(
            model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        train_metrics['loss'].append(train_loss)
        train_metrics['acc'].append(train_acc)
        train_metrics['precision'].append(train_precision)
        train_metrics['recall'].append(train_recall)
        train_metrics['f1'].append(train_f1)
        
        val_metrics['loss'].append(val_loss)
        val_metrics['acc'].append(val_acc)
        val_metrics['precision'].append(val_precision)
        val_metrics['recall'].append(val_recall)
        val_metrics['f1'].append(val_f1)
        

        epoch_time = time.time() - epoch_start
        print(f'Epoch: {epoch+1}/{args.epochs} | '
              f'Train Loss: {train_loss:.6f} | '
              f'Train Acc: {train_acc:.6f} | '
              f'Train F1: {train_f1:.6f} | '
              f'Val Loss: {val_loss:.6f} | '
              f'Val Acc: {val_acc:.6f} | '
              f'Val F1: {val_f1:.6f} | '
              f'Time: {epoch_time:.2f}s | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_precision = val_precision
            best_val_recall = val_recall
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            
            best_model_path = model_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'seed': seed,
                'data_split': split_info
            }, best_model_path)
            print(f"保存最佳模型到 {best_model_path}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= args.patience:
            print(f"验证集损失 {args.patience} 轮未下降，提前终止训练")
            break

    print(f"加载最佳模型 (Epoch {best_epoch+1})")
    checkpoint = torch.load(model_dir / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    

    test_acc, test_loss, test_precision, test_recall, test_f1, test_auc, y_true, y_pred, y_probs = evaluate(
        model, test_loader, criterion, device, return_probs=True)
    print(f"测试集结果: Loss: {test_loss:.6f}, Accuracy: {test_acc:.6f}, F1: {test_f1:.6f}, AUC: {test_auc:.6f}")
    
    # 绘制ROC曲线和混淆矩阵
    roc_auc_score = plot_roc_and_confusion_matrix(y_true, y_pred, y_probs, output_dir, "MGTAF")
    
    # 保存混淆矩阵数据
    cm_data = save_confusion_matrix_data(y_true, y_pred, output_dir)
    
    total_time = time.time() - start_time
    print(f"训练完成! 总耗时: {total_time:.2f}秒")

    metrics_data = {
        'train': train_metrics,
        'val': val_metrics
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    results = {
        "seed": seed,
        "train_size": len(training_set),
        "val_size": len(validation_set),
        "test_size": len(test_set),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "best_val_precision": float(best_val_precision),
        "best_val_recall": float(best_val_recall),
        "best_val_f1": float(best_val_f1),
        "best_epoch": best_epoch+1,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
        "roc_auc": float(roc_auc_score),
        "confusion_matrix": cm_data,
        "total_time": float(total_time)
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"随机种子: {seed}\n")
        f.write(f"训练集大小: {len(training_set)}\n")
        f.write(f"验证集大小: {len(validation_set)}\n")
        f.write(f"测试集大小: {len(test_set)}\n")
        f.write(f"最佳验证集损失: {best_val_loss:.6f} (Epoch {best_epoch+1})\n")
        f.write(f"最佳验证集准确率: {best_val_acc:.6f}\n")
        f.write(f"最佳验证集精确率: {best_val_precision:.6f}\n")
        f.write(f"最佳验证集召回率: {best_val_recall:.6f}\n")
        f.write(f"最佳验证集F1分数: {best_val_f1:.6f}\n")
        f.write(f"测试集损失: {test_loss:.6f}\n")
        f.write(f"测试集准确率: {test_acc:.6f}\n")
        f.write(f"测试集精确率: {test_precision:.6f}\n")
        f.write(f"测试集召回率: {test_recall:.6f}\n")
        f.write(f"测试集F1分数: {test_f1:.6f}\n")
        f.write(f"测试集AUC: {test_auc:.6f}\n")
        f.write(f"ROC AUC分数: {roc_auc_score:.6f}\n")
        f.write(f"混淆矩阵:\n")
        f.write(f"  真阴性(TN): {cm_data['true_negatives']}\n")
        f.write(f"  假阳性(FP): {cm_data['false_positives']}\n")
        f.write(f"  假阴性(FN): {cm_data['false_negatives']}\n")
        f.write(f"  真阳性(TP): {cm_data['true_positives']}\n")
        f.write(f"  特异性: {cm_data['specificity']:.6f}\n")
        f.write(f"  敏感性: {cm_data['sensitivity']:.6f}\n")
        f.write(f"总训练时间: {total_time:.2f}秒\n")
    
    return test_acc, best_val_acc, seed, split_info

#---------------------------------------------
# 主函数
#---------------------------------------------
def main():
    """主函数，处理多次训练运行并记录最佳结果"""
    args = parse_args()
    
    # 首先检查GPU状态
    gpu_available = check_gpu_status()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_results = []
    best_acc = 0
    best_seed = None
    best_run_id = None
    
    print(f"\n开始训练，使用固定种子 {args.seed}，共进行 {args.max_attempts} 次")
    print(f"目标准确率: {args.target_acc}")
    print(f"达到目标时停止: {'是' if args.stop_on_target else '否'}")
    print(f"每次使用相同种子: {args.seed}")
    
    # 如果是测试模式
    if args.test_loop:
        print("测试模式：检查循环是否正常工作")
        for i in range(min(5, args.max_attempts)):
            print(f"测试循环 {i+1}")
            import time
            time.sleep(0.5)  # 短暂暂停
        print("测试循环完成")
        return
    
    summary_file = output_dir / "summary.csv"
    with open(summary_file, "w") as f:
        f.write("run_id,seed,test_accuracy,val_accuracy,test_f1,test_precision,test_recall\n")
    
    base_seed = args.seed
    
    print(f"\n开始主循环，预计运行 {args.max_attempts} 次...")
    
    for run in range(args.max_attempts):
        print(f"\n{'='*60}")
        print(f"开始运行 {run+1}/{args.max_attempts}")
        print(f"{'='*60}")
        
        seed = base_seed
        print(f"使用种子: {seed} (基础种子: {base_seed}, 运行ID: {run+1})")
        
        try:
            print("调用 train_with_seed 函数...")
            test_acc, val_acc, used_seed, split_info = train_with_seed(args, seed, run)
            print(f"train_with_seed 函数执行完毕，返回: test_acc={test_acc}, val_acc={val_acc}")
            
        except KeyboardInterrupt:
            print(f"\n用户中断程序 (Ctrl+C)")
            break
        except Exception as e:
            print(f"运行 {run+1} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            print("继续下一次运行...")
            continue
        
        print("获取结果文件信息...")
        result_file = output_dir / f"run_{run:03d}_seed_{seed}" / "results.json"
        if result_file.exists():
            print(f"找到结果文件: {result_file}")
            with open(result_file, "r") as f:
                result_data = json.load(f)
                test_f1 = result_data.get("test_f1", 0)
                test_precision = result_data.get("test_precision", 0)
                test_recall = result_data.get("test_recall", 0)
        else:
            print(f"结果文件不存在: {result_file}")
            test_f1 = 0
            test_precision = 0
            test_recall = 0
        
        result = {
            "run_id": run+1,
            "seed": used_seed,
            "test_accuracy": test_acc,
            "val_accuracy": val_acc,
            "test_f1": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall
        }
        all_results.append(result)
        print(f"结果已添加到列表，当前结果数量: {len(all_results)}")
        
        print("更新CSV文件...")
        with open(summary_file, "a") as f:
            f.write(f"{run+1},{used_seed},{test_acc:.6f},{val_acc:.6f},{test_f1:.6f},{test_precision:.6f},{test_recall:.6f}\n")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_seed = used_seed
            best_run_id = run+1
            print(f"发现新的最佳准确率!")
            
        print(f"当前最佳准确率: {best_acc:.6f} (种子: {best_seed}, 运行: {best_run_id})")
        
        # 如果启用了达到目标时停止，并且已达到目标准确率
        if args.stop_on_target and best_acc >= args.target_acc:
            print(f"已达到目标准确率 {args.target_acc:.6f}，提前终止训练")
            break
            
        # 添加进度信息
        print(f"已完成运行: {run+1}/{args.max_attempts}")
        remaining = args.max_attempts - (run+1)
        print(f"剩余运行次数: {remaining}")
        
        # 清理内存
        print("清理内存...")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 打印GPU内存使用情况
            if torch.cuda.current_device() >= 0:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_cached = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU内存 - 已分配: {memory_allocated:.2f} GB, 缓存: {memory_cached:.2f} GB")
        
        print(f"运行 {run+1} 完成，准备下一次运行...")
        
        # 检查是否是最后一次运行
        if run + 1 == args.max_attempts:
            print("这是最后一次运行")
        else:
            print(f"准备开始第 {run+2} 次运行...")
    
    # 循环结束后的处理
    print(f"\n{'='*60}")
    print("主循环已结束")
    print(f"循环执行了 {len(all_results)} 次")
    print(f"{'='*60}")
    
    print(f"\n=== 训练完成 ===")
    print(f"最佳准确率: {best_acc:.6f}")
    print(f"最佳种子: {best_seed}, 运行ID: {best_run_id}")
    print(f"目标准确率: {args.target_acc}, {'已达到' if best_acc >= args.target_acc else '未达到'}")
    
    results_summary = {
        "base_seed": args.seed,
        "target_accuracy": args.target_acc,
        "best_accuracy": float(best_acc),
        "best_val_accuracy": float(all_results[best_run_id-1]["val_accuracy"]) if best_run_id is not None else None,
        "best_seed": int(best_seed) if best_seed is not None else None,
        "best_run_id": best_run_id,
        "total_runs": args.max_attempts,
        "completed_runs": len(all_results),
        "success": best_acc >= args.target_acc,
        "all_results": all_results
    }
    
    with open(output_dir / "final_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"所有结果已保存到 {output_dir}")

def plot_roc_and_confusion_matrix(y_true, y_pred, y_probs, output_dir, model_name="MGTAF"):
    """绘制ROC曲线和混淆矩阵"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建一个包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_true, np.array(y_probs)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'{model_name} ')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix')
    ax2.set_xticklabels(['NC', 'ASD'])
    ax2.set_yticklabels(['NC', 'ASD'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'roc_confusion_matrix.pdf', bbox_inches='tight')
    plt.close()
    
    return roc_auc

def save_confusion_matrix_data(y_true, y_pred, output_dir):
    """保存混淆矩阵数据"""
    cm = confusion_matrix(y_true, y_pred)
    cm_data = {
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1]),
        'specificity': cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0,
        'sensitivity': cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    }
    
    with open(Path(output_dir) / 'confusion_matrix_data.json', 'w') as f:
        json.dump(cm_data, f, indent=2)
    
    return cm_data

if __name__ == '__main__':
    main()