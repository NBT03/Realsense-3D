import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO

def initialize_camera():
    """Khởi tạo kết nối với camera RealSense D435i"""
    # Tạo pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Bật stream màu và độ sâu
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Bắt đầu pipeline
    pipeline.start(config)
    
    return pipeline

def get_aligned_frames(pipeline):
    """Lấy và căn chỉnh khung hình màu và độ sâu"""
    # Đợi một bộ khung hình
    frames = pipeline.wait_for_frames()
    
    # Tạo alignment object
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    
    # Lấy khung hình đã căn chỉnh
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        return None, None
    
    # Chuyển đổi khung hình thành mảng numpy
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    return color_image, depth_image

def process_depth(depth_image):
    """Xử lý ảnh độ sâu để hiển thị"""
    # Áp dụng colormap cho ảnh độ sâu
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    return depth_colormap

def main():
    # Khởi tạo camera
    pipeline = initialize_camera()
    
    # Tải model YOLOv11 với khả năng segmentation
    model = YOLO('best.pt')  # Thay thế bằng đường dẫn chính xác đến model
    
    try:
        while True:
            # Lấy khung hình từ camera
            color_image, depth_image = get_aligned_frames(pipeline)
            if color_image is None or depth_image is None:
                continue
                
            # Tạo bản sao của ảnh để vẽ lên
            display_image = color_image.copy()
            
            # Chạy phát hiện và phân đoạn với YOLOv11
            results = model(color_image, task='segment')
            
            # Xử lý kết quả
            for result in results:
                # Lấy masks nếu có
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    
                    # Lấy thông tin boxes và classes
                    boxes = result.boxes.data.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    # Xử lý từng vật thể được phát hiện
                    for i, (mask, box, cls, conf) in enumerate(zip(masks, boxes, classes, confs)):
                        if conf < 0.5:  # Lọc theo độ tin cậy
                            continue
                            
                        # Chuyển đổi mask thành định dạng hình ảnh
                        seg_mask = mask.astype(np.uint8)
                        seg_mask = cv2.resize(seg_mask, (color_image.shape[1], color_image.shape[0]))
                        
                        # Tạo mask màu
                        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                        colored_mask = np.zeros_like(color_image, dtype=np.uint8)
                        colored_mask[seg_mask > 0] = color
                        
                        # Kết hợp mask với ảnh hiển thị
                        alpha = 0.5
                        display_image = cv2.addWeighted(display_image, 1, colored_mask, alpha, 0)
                        
                        # Vẽ bbox
                        x1, y1, x2, y2 = box[:4].astype(int)
                        cv2.rectangle(display_image, (x1, y1), (x2, y2), color.tolist(), 2)
                        
                        # Lấy tên lớp và độ tin cậy
                        class_name = model.names[int(cls)]
                        label = f'{class_name} {conf:.2f}'
                        
                        # Tính toán độ sâu trung bình trong vùng mask
                        masked_depth = depth_image.copy()
                        masked_depth[seg_mask == 0] = 0
                        non_zero_depths = masked_depth[masked_depth > 0]
                        if len(non_zero_depths) > 0:
                            avg_depth = np.mean(non_zero_depths)
                            label += f' - {avg_depth:.0f}mm'
                        
                        # Hiển thị nhãn
                        cv2.putText(display_image, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)
            
            # Xử lý ảnh độ sâu
            depth_colormap = process_depth(depth_image)
            
            # Hiển thị kết quả
            cv2.imshow('Segmentation Result', display_image)
            cv2.imshow('Depth Map', depth_colormap)
            
            # Thoát khi nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Dọn dẹp
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
