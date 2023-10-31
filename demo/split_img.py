import cv2

def draw_grid_on_image(image_path, M, N, output_path):
    # 读取图片
    image = cv2.imread(image_path)

    # 获取图片的高度和宽度
    height, width, _ = image.shape

    # 计算每个块的大小
    block_height = height // M
    block_width = width // N

    # 画纵线
    for i in range(1, N):
        start_point = (i * block_width, 0)
        end_point = (i * block_width, height)
        cv2.line(image, start_point, end_point, (0, 0, 255), 1)  # 红色线条

    # 画横线
    for i in range(1, M):
        start_point = (0, i * block_height)
        end_point = (width, i * block_height)
        cv2.line(image, start_point, end_point, (0, 0, 255), 1)  # 红色线条

    # 保存图片
    cv2.imwrite(output_path, image)

# 使用示例
image_path = '/home/wiser-renjie/remote_datasets/yolov5_images/highway/000001.jpg'
output_path = '/home/wiser-renjie/remote_datasets/yolov5_images/highway/000001_split.jpg'
M, N = 10, 20  # 将图片分割成5x5的块
draw_grid_on_image(image_path, M, N, output_path)
