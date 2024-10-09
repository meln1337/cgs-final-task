import os
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN
from utils import load_ce_efficientnet, \
    load_arface_efficientnet, load_ce_inceptionresnetv1, load_triplet_efficientnet, encode, align_face
from config import params_dict
from PIL import Image
from types import MethodType
import argparse

model_names = ['arcface-efficientnet', 'ce-efficientnet',
               'ce-inceptionresnetv1-1292', 'triplet-efficientnet-last-epoch',
               'triplet-efficientnet-min-val-loss']

parser = argparse.ArgumentParser(
    prog='Real-time face recognition',
    description='Face recognition',
    epilog='Specify needed parameters'
)
parser.add_argument("-model_name",
                    type=str, default='ce-inceptionresnetv1-1292',
                    help=f"Name of the model, available: {model_names}")
parser.add_argument("-device",
                    type=str, default='cpu',
                    help=f"Device to compute (cpu or cuda)")
parser.add_argument("-cam",
                    type=int, default=0,
                    help=f"Camera to capture")
parser.add_argument('--landmarks', action=argparse.BooleanOptionalAction)
# parser.add_argument('--no-landmarks', dest='feature', action='store_false')
parser.add_argument("-people_faces",
                    type=str, default='people_faces',
                    help=f"Folder which contains faces of people to detect")
parser.add_argument("-output_path",
                    type=str, default='gifs/detected.gif',
                    help=f"Path for saving detected gif")


args = parser.parse_args()
model_name = args.model_name
device = torch.device(args.device)

print(f'Using model: {model_name} on {device}, camera: {args.cam}, output_path: {args.output_path}, '
      f'landmarks: {args.landmarks}')

if model_name not in model_names:
    print(f'{model_name} is not present in available models. Please choose one from: {model_names}')
    exit()

def detect_box(self, img, save_path=None):
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    if args.landmarks:
        return batch_boxes, faces, batch_points
    else:
        return batch_boxes, faces, None

### load model

if model_name in 'arcface-efficientnet':
    model = load_arface_efficientnet(params_dict[model_name]['path'], 1292)
elif model_name == 'ce-efficientnet':
    model = load_ce_efficientnet(params_dict[model_name]['path'], 1292)
elif model_name == 'ce-inceptionresnetv1-1292':
    model = load_ce_inceptionresnetv1(params_dict[model_name]['path'], 1292)
elif model_name in ['triplet-efficientnet-last-epoch', 'triplet-efficientnet-min-val-loss']:
    model = load_triplet_efficientnet(params_dict[model_name]['path'], 1292)

# print(model)
model = model.eval()

mtcnn = MTCNN(
  image_size=160, keep_all=True, thresholds=[0.65, 0.75, 0.85], min_face_size=60
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

### get encoded features for all people_faces images
images_path = 'people_faces'
all_people_faces = {}

for file in os.listdir(os.path.join(args.people_faces)):
    img = cv2.imread(os.path.join(args.people_faces, file))
    batch_boxes, cropped_images, batch_landmarks = mtcnn.detect_box(img)

    if cropped_images is not None:
        cropped_images = np.array([align_face(
            img.numpy().transpose(1, 2, 0), landmark) for img, landmark in zip(cropped_images, batch_landmarks)
        ])

        cropped_images = cropped_images.transpose(0, 3, 1, 2)

        all_people_faces[file] = encode(model_name, model, torch.from_numpy(cropped_images))[0]

print('Used MTCNN on people_faces images')

gif_files = []

def detect(cam=0, thres=params_dict[model_name]['threshold']):
    vdo = cv2.VideoCapture(cam)
    while vdo.grab():
        _, img0 = vdo.retrieve()
        batch_boxes, cropped_images, batch_landmarks = mtcnn.detect_box(img0)

        if cropped_images is None:
            print('No images has been detected')
            continue

        cropped_images = [align_face(np.array(img.permute(2, 0, 1)), landmark) for img, landmark in zip(cropped_images, batch_landmarks)]


        if batch_landmarks is not None:
            for landmarks in batch_landmarks:
                # Loop through each face's landmarks
                for (x, y) in landmarks:
                    # Draw each landmark as a small circle
                    cv2.circle(img0, (int(x), int(y)), 5, (0, 255, 0), -1)

        if cropped_images is not None:
            for i, (box, cropped) in enumerate(zip(batch_boxes, cropped_images)):
                if cropped is None or box is None:
                    continue
                x, y, x2, y2 = [int(x) for x in box]

                img_embedding = encode(model_name, model, torch.from_numpy(cropped).unsqueeze(0))
                detect_dict = {}
                for k, v in all_people_faces.items():
                    detect_dict[k] = (v - img_embedding).norm().item()

                print(detect_dict)
                min_key = min(detect_dict, key=detect_dict.get)

                if detect_dict[min_key] >= thres:
                    min_key = 'Undetected'

                print(f'score for face {i + 1}, recognized as {min_key}: {detect_dict[min_key]}')
                
                cv2.rectangle(img0, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                  img0, min_key, (x + 5, y + 10), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                gif_files.append(Image.fromarray(img0[:, :, ::-1]))
        else:
            gif_files.append(Image.fromarray(img0[:, :, ::-1]))
                
        ### display
        cv2.imshow("face recognition", img0)
        if cv2.waitKey(1) == ord('q'):
            # for gif_file in gif_files:
            cv2.destroyAllWindows()
            gif_files[0].save(args.output_path,
                           save_all=True,
                           append_images=gif_files[1:],
                           duration=100,  # time between frames in milliseconds
                           loop=0)
            break

if __name__ == "__main__":
    detect(args.cam)