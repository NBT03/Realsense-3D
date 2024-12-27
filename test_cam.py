import pyrealsense2 as rs
import numpy as np
import cv2

def test_realsense_camera():
    try:
        # Khởi tạo pipeline
        pipeline = rs.pipeline()
        
        # Cấu hình stream
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Bắt đầu stream
        pipeline.start(config)
        print("Camera started successfully. Press 'q' to exit.")

        while True:
            # Lấy frame từ pipeline
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Chuyển frame sang định dạng numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Chuẩn hóa hình ảnh độ sâu để hiển thị
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Kết hợp hình ảnh màu và hình ảnh độ sâu
            images = np.hstack((color_image, depth_colormap))

            # Hiển thị hình ảnh
            cv2.imshow('RealSense Camera', images)

            # Thoát bằng phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Dừng pipeline và đóng tất cả cửa sổ
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_realsense_camera()
