from .core_processing import *


def main():
    image_path = r'Images\darkimage.tif'
    dark_color_image = cv2.imread(image_path)
    dark_gray_image = cv2.cvtColor(dark_color_image, cv2.COLOR_BGR2GRAY)

    print("Running Script...\n")

    min_gray_value = int((2e0 - 1) - 1)  # Equivalent to 1 bit of data (Exponential representation)
    max_gray_value = int(2e2 + 5e1 + 6e0 - 1)  # Equivalent to 8 bits of data (Exponential representation)
    desired_range = [min_gray_value, max_gray_value]
    max_enhanced_image, scale_factor, bias_factor = contrast_enhancement(dark_gray_image, desired_range)

    # display images
    print('--- Displaying Images ---')
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(dark_gray_image)
    plt.title('Original')

    plt.subplot(1, 2, 2)
    plt.imshow(max_enhanced_image, cmap='gray', vmin=min_gray_value, vmax=max_gray_value)
    plt.title('Enhanced Contrast')

    # Print scale_factor,bias_factor
    print(f"Image Scale Factor = {scale_factor}, Image Bias Factor = {bias_factor}\n")

    # Display the Tone Mapping
    image_min_range = np.min(dark_gray_image)
    image_max_range = np.max(dark_gray_image)
    image_range = [image_min_range, image_max_range]
    show_image_mapping(image_range, scale_factor, bias_factor)

    print("'--- Bias Factor ---'\n")
    enhanced_2nd_degree_image, scale_factor, bias_factor = contrast_enhancement(max_enhanced_image, desired_range)
    print("Enhancing an already enhanced image...\n")
    print(f"Image Scale_factor = {scale_factor}, Bias_factor = {bias_factor}\n")

    print('Displaying the difference between the two enhanced factor images...\n')
    print('--- Displaying Images ---\n')
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(max_enhanced_image, cmap='gray', vmin=min_gray_value, vmax=max_gray_value)
    plt.title('Enhanced Image')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_2nd_degree_image, cmap='gray', vmin=min_gray_value, vmax=max_gray_value)
    plt.title('Second Stage Enhanced Image')

    image_difference: float = mean_square_distance(max_enhanced_image, enhanced_2nd_degree_image)
    print(f'The difference value (in terms of mean square error distance) between the two images is: '
          f'{image_difference}\n')

    print(" --- Minkowski p=2 Distance ---\n")
    minkowski_dist = minkowski_2_distance(dark_gray_image, dark_gray_image)
    print(f"Minkowski distance between image and itself is: {minkowski_dist}\n")

    # implement the loop that calculates minkowski distance as function of increasing contrast
    number_of_steps = 20
    step_size = (image_max_range - image_min_range) // 20
    minkowski_dists = np.zeros(number_of_steps)
    for step in range(number_of_steps):
        increasing_range = [image_min_range, (image_min_range + ((step+1) * step_size))]
        enhanced_step_image, _, _ = contrast_enhancement(dark_gray_image, increasing_range)
        minkowski_dists[step]: float = minkowski_2_distance(dark_gray_image, enhanced_step_image)

    image_max_contrast_step_range = image_min_range + number_of_steps * step_size
    contrast = np.arange(image_min_range, image_max_contrast_step_range, step_size)

    plt.figure()
    plt.plot(contrast, minkowski_dists)
    plt.xlabel("Contrast")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski distance as function of contrast steps")

    print('\n')
    print("--- Slice Matrix ---\n")

    tone_mapping_vector = np.arange(256)  # Tone Mapping vector of shape (1 x 256)
    reconstructed_image: np.ndarray = mapping_image(dark_gray_image, tone_mapping_vector)
    # Computationally Proving that sliceMat(image) * [0:255] == image
    image_slice_matrix_distance: float = mean_square_distance(dark_gray_image, reconstructed_image)
    print(f"The distance between the image and it's sliced reconstructed image is: {image_slice_matrix_distance}\n")

    # Displaying both images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(dark_gray_image)
    plt.title('Dark Gray Image')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray', vmin=min_gray_value, vmax=max_gray_value)
    plt.title('Reconstructed Sliced Masked Image')

    print("--- Tone Mapped Image Contrast Enhancement ---\n")

    # sliced_image_matrix_matrix: np.ndarray = slice_matrix(dark_gray_image)
    _, tone_mapping_vector = slt_map(dark_gray_image, max_enhanced_image)
    tone_mapped_image = mapping_image(dark_gray_image, tone_mapping_vector)  # Calculates sliced_matrix(image)*TM_vector
    enhanced_image_vs_tone_mapped_distance = mean_square_distance(max_enhanced_image, tone_mapped_image)
    print(f"Sum of difference between image and slices*[0..255] is: {enhanced_image_vs_tone_mapped_distance}")

    # Displaying Images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(dark_color_image)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(tone_mapped_image, cmap='gray', vmin=0, vmax=255)
    plt.title("Tone Mapped Image")

    print('\n')
    print("--- SLT Negative Image ---\n")
    negative_image = slt_negative(dark_gray_image)
    plt.figure()
    plt.imshow(negative_image, cmap='gray', vmin=0, vmax=255)
    plt.title("Negative Image Using SLT")

    print("--- SLT Threshold ---")
    threshold = 120
    lena = cv2.imread(r"Images\\RealLena.tif")
    lena_gray_image = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    thresholded_image = slt_threshold(lena_gray_image, threshold)

    # Computationally testing the thresholded image correctness
    binary_threshold_image = 1 * (lena_gray_image > threshold)
    distance_threshold_image = mean_square_distance(thresholded_image, 255 * binary_threshold_image)
    print(f'The difference between the thresholded image by SLT method, and the ground truth thresholded image is: '
          f'{distance_threshold_image}\n')

    plt.figure()
    plt.imshow(thresholded_image, cmap='gray', vmin=0, vmax=255)
    plt.title("Thresholded Image Using SLT")

    print("--- SLT Mapping ---\n")
    image_1 = lena_gray_image
    image_2 = dark_gray_image
    slt_image_1, _ = slt_map(lena_gray_image, dark_gray_image)

    # Displaying Images
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image_1)
    plt.title("Original image")
    plt.subplot(1, 3, 2)
    plt.imshow(slt_image_1, cmap='gray', vmin=0, vmax=255)
    plt.title("Tone Mapped Image")
    plt.subplot(1, 3, 3)
    plt.imshow(image_2, cmap='gray', vmin=0, vmax=255)
    plt.title("Target Image for Tone Mapping")

    image1_image2_distance = mean_square_distance(image_1, image_2)  # Mean square distance between image_1 and image_2

    # Mean square distance between tone mapped image_1 and image_2
    tone_mapped_image1_image2_distance = mean_square_distance(slt_image_1, image_2)
    print(f"Mean square distance between image_1 and image_2 is: {image1_image2_distance}")
    print(f"Mean square distance between tone mapped image_1 and image_2 is: {tone_mapped_image1_image2_distance}")
    print(f"Hence, D(im1, im2) = {image1_image2_distance} > {tone_mapped_image1_image2_distance} = D(TM_im1, im2)")

    print("--- Symmetric SLT Map ---\n")

    # Computationally Proving
    slt_image_2, _ = slt_map(image_2, image_1)
    slt_symmetric_map_distance = mean_square_distance(slt_image_1, slt_image_2)
    print(f"The Mean Square Distance between symmetric transformations is: {slt_symmetric_map_distance}")

    # Displaying both images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(slt_image_1, cmap='gray', vmin=0, vmax=255)
    plt.title("Tone Mapped Image 1 -> Image 2")
    plt.subplot(1, 2, 2)
    plt.imshow(slt_image_2, cmap='gray', vmin=0, vmax=255)
    plt.title("Tone Mapped Image 2 -> Image 1")
    plt.show()


if __name__ == "__main__":

    main()
