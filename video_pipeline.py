def perform_binary_warp(input_image):
    # Use input_image for perspective transform
    input_image_orig = np.copy(input_image)
    src_points_array = np.array([[550, 480], [750, 480], [1100, 700], [320, 700]])
    dst_points_array = np.array([[200, 100], [1100, 100], [1100, 650], [200, 650]])

    src_points = np.int32(src_points_array)
    dst_points = np.int32(dst_points_array)
    src_points = src_points.reshape((-1,1,2))
    dst_points = dst_points.reshape((-1,1,2))
    cv2.polylines(input_image, [src_points], True, (20, 20, 20), 2)
    cv2.polylines(input_image, [dst_points], True, (20, 20, 20), 2)
    print_gray_image(input_image)

    # start warping the image
    src_points = np.float32(src_points_array)
    dst_points = np.float32(dst_points_array)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    img_size = (input_image_orig.shape[1], input_image_orig.shape[0])
    binary_warped = cv2.warpPerspective(input_image_orig, M, img_size)
    return binary_warped, M, Minv
