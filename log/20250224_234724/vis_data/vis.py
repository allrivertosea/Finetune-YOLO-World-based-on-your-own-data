import json
from torch.utils.tensorboard import SummaryWriter

# 创建 SummaryWriter 对象
writer = SummaryWriter(log_dir='logs')

# 打开并读取 JSON 文件
with open(r'D:\github-demo\yoloworldbase\log\20250224_234724\vis_data\20250224_234724.json', 'r') as f:
    # 遍历每行 JSON 数据
# 遍历每行 JSON 数据
    for line in f:
        # 解析 JSON 数据
        entry = json.loads(line.strip())
        
        # 打印 entry 内容，检查是否包含需要的字段
        
        # 记录损失相关的指标
        if 'loss' in entry and 'iter' in entry:
            writer.add_scalar('Loss/train', entry['loss'], entry['iter'])
            writer.add_scalar('Loss/train_cls', entry['loss_cls'], entry['iter'])
            writer.add_scalar('Loss/train_bbox', entry['loss_bbox'], entry['iter'])
            writer.add_scalar('Loss/train_dfl', entry['loss_dfl'], entry['iter'])
            writer.add_scalar('Learning_Rate', entry['lr'], entry['iter'])
            writer.add_scalar('Grad_Norm', entry['grad_norm'], entry['iter'])
            writer.add_scalar('Memory', entry['memory'], entry['iter'])
            writer.add_scalar('Epoch', entry['epoch'], entry['iter'])
        
        # 检查是否包含 'step' 字段（这就是我们迭代次数的标记）
        if 'step' in entry:
            # 记录 COCO mAP 指标到 TensorBoard
            if 'coco/bbox_mAP' in entry:
                writer.add_scalar('COCO/bbox_mAP', entry['coco/bbox_mAP'], entry['step'])
            if 'coco/bbox_mAP_50' in entry:
                writer.add_scalar('COCO/bbox_mAP_50', entry['coco/bbox_mAP_50'], entry['step'])
            if 'coco/bbox_mAP_75' in entry:
                writer.add_scalar('COCO/bbox_mAP_75', entry['coco/bbox_mAP_75'], entry['step'])
            if 'coco/bbox_mAP_s' in entry:
                writer.add_scalar('COCO/bbox_mAP_s', entry['coco/bbox_mAP_s'], entry['step'])
            if 'coco/bbox_mAP_m' in entry:
                writer.add_scalar('COCO/bbox_mAP_m', entry['coco/bbox_mAP_m'], entry['step'])
            if 'coco/bbox_mAP_l' in entry:
                writer.add_scalar('COCO/bbox_mAP_l', entry['coco/bbox_mAP_l'], entry['step'])

            # 记录其他可能的指标
            if 'data_time' in entry:
                writer.add_scalar('Data_Time', entry['data_time'], entry['step'])
            if 'time' in entry:
                writer.add_scalar('Time', entry['time'], entry['step'])

# 关闭 writer
writer.close()
