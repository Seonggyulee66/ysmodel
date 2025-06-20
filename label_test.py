import matplotlib.pyplot as plt
import numpy as np
import cv2

image_path = 'E:/DATA_train/merge_image_result/train/2021_08_16_22_26_54/641/000069_merged_bev.png'

image = plt.imread(image_path)

def generate_label(bev_map):
        """
        Convert rgb images to binary output.

        Parameters
        ----------
        bev_map : np.ndarray
            Uint 8 image with 3 channels.
        """
        bev_map = cv2.cvtColor(bev_map, cv2.COLOR_BGR2GRAY)
        bev_map = np.array(bev_map, dtype=np.float64) / 255.
        bev_map[bev_map > 0] = 1

        return bev_map


image_post_processed = generate_label(image)

# 시각화 (원본 + 변환 결과)
plt.figure(figsize=(10, 5))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# 전처리된 이미지
plt.subplot(1, 2, 2)
plt.imshow(image_post_processed, cmap='gray')
plt.title('Post Processed Image')
plt.axis('off')

plt.tight_layout()
plt.show()