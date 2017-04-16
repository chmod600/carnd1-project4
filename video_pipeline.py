global current_image, binary_color_combined, warped_binary, M, Minv, visualization_image, left_fit, right_fit, left_lane_inds, right_lane_inds

def process_image(image):
    current_image = image
    extract_combined_binary()
    perform_binary_warp()
    fit_polynomial()
    measure_curvature()
    project_lane_on_image()

def extract_combined_binary():
    ksize = 15
    gradx = abs_sobel_thresh(current_image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(current_image, orient='y', sobel_kernel=ksize, thresh=(20, 100))

    mag_binary = mag_thresh(current_image, sobel_kernel = ksize, mag_thresh=(25, 100))
    dir_binary = dir_threshold(current_image, sobel_kernel = 15, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    binary_color_combined = np.zeros_like(combined)
    s_binary = hls_select(current_image, thresh=(90, 200))
    binary_color_combined[((combined == 1) | (s_binary == 1))] = 1

def perform_binary_warp(binary_image):
    # Use input_image for perspective transform
    binary_image_orig = np.copy(binary_image)
    src_points_array = np.array([[550, 480], [750, 480], [1100, 700], [320, 700]])
    dst_points_array = np.array([[200, 100], [1100, 100], [1100, 650], [200, 650]])

    src_points = np.int32(src_points_array)
    dst_points = np.int32(dst_points_array)

    src_points = src_points.reshape((-1,1,2))
    dst_points = dst_points.reshape((-1,1,2))

    cv2.polylines(binary_image, [src_points], True, (20, 20, 20), 2)
    cv2.polylines(binary_image, [dst_points], True, (20, 20, 20), 2)

    print_gray_image(binary_image, "Perspective Transform mappings")

    # start warping the image
    src_points = np.float32(src_points_array)
    dst_points = np.float32(dst_points_array)

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)

    img_size = (binary_image_orig.shape[1], binary_image_orig.shape[0])
    binary_warped = cv2.warpPerspective(binary_image_orig, M, img_size)
    return binary_warped, M, Minv



binary_color_combined_orig = np.copy(binary_color_combined)
src_points_array = np.array([[550, 480], [750, 480], [1100, 700], [320, 700]])
dst_points_array = np.array([[200, 100], [1100, 100], [1100, 650], [200, 650]])

src_points = np.int32(src_points_array)
dst_points = np.int32(dst_points_array)
src_points = src_points.reshape((-1,1,2))
dst_points = dst_points.reshape((-1,1,2))
cv2.polylines(binary_color_combined, [src_points], True, (20, 20, 20), 2)
cv2.polylines(binary_color_combined, [dst_points], True, (20, 20, 20), 2)
print_gray_image(binary_color_combined)

# start warping the image
src_points = np.float32(src_points_array)
dst_points = np.float32(dst_points_array)

M = cv2.getPerspectiveTransform(src_points, dst_points)
Minv = cv2.getPerspectiveTransform(dst_points, src_points)
img_size = (binary_color_combined_orig.shape[1], binary_color_combined_orig.shape[0])
binary_warped = cv2.warpPerspective(binary_color_combined_orig, M, img_size)
print_gray_image(binary_warped)
