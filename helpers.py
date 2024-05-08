import cv2
import cv2
import numpy as np
import scipy.interpolate
import sys
import matplotlib.pyplot as plt




def detect_features(image_paths):

    detector = cv2.SIFT_create()

    keypoints_all = []
    descriptors_all = []

    for path in image_paths:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        keypoints, descriptors = detector.detectAndCompute(image, None)
        keypoints_all.append(keypoints)
        descriptors_all.append(descriptors)

    return keypoints_all, descriptors_all

def draw_keypoints(image_paths, keypoints_list):

    fig, axes = plt.subplots(nrows=len(image_paths), ncols=1, figsize=(10, 10 * len(image_paths)))
    if len(image_paths) == 1:
        axes = [axes]

    for ax, image_path, keypoints in zip(axes, image_paths, keypoints_list):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(im_keypoints)
        ax.axis('off')

    plt.show()



def match_features(descriptors1, descriptors2, crossCheck=True):

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    matches = bf.match(descriptors1, descriptors2)
    return sorted(matches, key=lambda x: x.distance)

def find_good_matches_sequential(descriptors_list, good_match_percentage=0.30):

    good_matches = []
    for i in range(len(descriptors_list) - 1):
        matches = match_features(descriptors_list[i], descriptors_list[i+1])
        good_matches.append(matches[:int(len(matches) * good_match_percentage)])
    return good_matches

def find_good_matches_base(descriptors_list, base_image_index, good_match_percentage=0.50):

    good_matches = []
    for i in range(base_image_index, len(descriptors_list) - 1):
        matches = match_features(descriptors_list[base_image_index], descriptors_list[i+1])
        good_matches.append(matches[:int(len(matches) * good_match_percentage)])
    return good_matches

def build_tracks(matches_list, keypoints_list):

    tracks = [[(0, m.queryIdx), (1, m.trainIdx)] for m in matches_list[0]]

    for i in range(1, len(matches_list)):
        new_tracks = []
        current_matches = matches_list[i]
        keypoint_to_track_map = {track[-1][1]: track for track in tracks if track[-1][0] == i}

        for match in current_matches:
            previous_keypoint_idx = match.queryIdx 
            current_keypoint_idx = match.trainIdx   

            if previous_keypoint_idx in keypoint_to_track_map:
                # Extend the existing track
                track = keypoint_to_track_map[previous_keypoint_idx]
                track.append((i+1, current_keypoint_idx))
            else:
                new_tracks.append([(i, previous_keypoint_idx), (i+1, current_keypoint_idx)])

        tracks.extend(new_tracks)

    return tracks

def draw_base_matches(image_paths, matches_list, keypoints_list, base_image_index):
    """
    Draw matches between consecutive image pairs.
    """
    for i in range(base_image_index, len(image_paths) - 1):

        img1 = cv2.imread(image_paths[base_image_index], cv2.IMREAD_COLOR)
        img2 = cv2.imread(image_paths[i+1], cv2.IMREAD_COLOR)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        keypoints1 = keypoints_list[base_image_index]
        keypoints2 = keypoints_list[i+1]

        img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches_list[i], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


        plt.figure(figsize=(12, 6))
        plt.axis(False)
        plt.imshow(img_matches)
        plt.title(f'Matches between image {base_image_index} and image {i+1}')
        plt.show()

def draw_sequential_matches(image_paths, matches_list, keypoints_list):

    for i in range(len(image_paths) - 1):
        img1 = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        img2 = cv2.imread(image_paths[i+1], cv2.IMREAD_COLOR)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        keypoints1 = keypoints_list[i]
        keypoints2 = keypoints_list[i+1]

        img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches_list[i], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(12, 6))
        plt.axis(False)
        plt.imshow(img_matches)
        plt.title(f'Matches between image {i} and image {i+1}')
        plt.show()


def draw_global_matches(image_paths, tracks, keypoints_list):
    """
    Draw matches that are present across all images.
    """
    num_images = len(image_paths)

    full_tracks = [track for track in tracks if len(track) == num_images]

    images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths]

    plt.figure()

    for i, img in enumerate(images):
        plt.subplot(num_images, 1, i+1)
        for track in full_tracks:
            img_index, kp_index = track[i]
            x, y = keypoints_list[img_index][kp_index].pt
            plt.scatter(x, y, color='red')

        plt.imshow(img)
        plt.axis(False)
        plt.title(f'Image {i}')
    plt.suptitle('Matches Across all Images')
    plt.tight_layout()
    plt.show()
def draw_matches(image_paths, keypoints_list, matches_list):
    """
    Draw matches and show the images.
    """
    for i in range(len(matches_list)):
        img1 = cv2.cvtColor(cv2.imread(image_paths[i]), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(image_paths[i+1]), cv2.COLOR_BGR2RGB)
        img_matches = cv2.drawMatches(img1, keypoints_list[i], img2, keypoints_list[i+1],
                                      matches_list[i], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(6, 6))
        plt.imshow(img_matches)
        plt.axis('off')
        plt.show()

def find_good_matches(descriptors_list, good_match_percentage=0.50):
    """
    Find good matches between consecutive image pairs in a list of descriptors.
    """
    good_matches = []
    for i in range(len(descriptors_list) - 1):
        matches = match_features(descriptors_list[i], descriptors_list[i+1])
        good_matches.append(matches[:int(len(matches) * good_match_percentage)])
    return good_matches


