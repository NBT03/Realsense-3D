import pyrealsense2 as rs
import numpy as np
import cv2

def get_3d_coordinates():
    try:
        # Khởi tạo pipeline
        pipeline = rs.pipeline()

        # Cấu hình stream
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Bắt đầu stream
        pipeline.start(config)
        print("Camera started. Press 'q' to exit.")

        while True:
            # Lấy frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Chuyển frame sang numpy array
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Lấy thông tin intrinsics của depth
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            # Hiển thị hình ảnh màu
            cv2.imshow("Color Frame", color_image)

            # Nhấn chuột để chọn vị trí pixel
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    # Lấy giá trị độ sâu tại pixel (x, y)
                    depth = depth_frame.get_distance(x, y)
                    if depth > 0:
                        # Tính tọa độ 3D
                        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
                        print(f"Pixel ({x}, {y}) - Depth: {depth}m - 3D Coordinates: {point_3d}")
                    else:
                        print(f"Pixel ({x}, {y}) has no depth data.")

            cv2.setMouseCallback("Color Frame", mouse_callback)

            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Dừng pipeline và đóng cửa sổ
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    get_3d_coordinates()
