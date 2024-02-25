import numpy as np
import torch
import tensorflow as tf
import torchvision.transforms as transforms
from skimage.transform import resize
from torchvision.ops import masks_to_boxes
from keras.layers import Dropout, Conv2D, MaxPooling2D, Input, UpSampling2D, Concatenate
from keras.models import Model


def crop_optical_dics(image, crop_model1, crop_model2, crop_model3, crop_model4):

    def tf_to_th_encoding(X):
        return np.rollaxis(X, 3, 1)

    def th_to_tf_encoding(X):
        return np.rollaxis(X, 1, 4)

    with tf.device('/GPU:0'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512))
        ])
        # transform_128 = transforms.Compose([
        #     transforms.Resize((128, 128))
        # ])
        # image = transform(image)
        im = image.detach().cpu().numpy()
        im = np.transpose(im, (0, 2, 3, 1))
        # im = im.astype(np.float64) / 255.0
        cropped_images = []
        for index, image in enumerate(im):
            w, h, c = image.shape
            image = np.expand_dims(image, axis=0)
            image_128 = resize(image, (1, 128, 128, 3), anti_aliasing=True)
            image = np.transpose(image, (0, 3, 1, 2))
            image_128 = np.transpose(image_128, (0, 3, 1, 2))
            try:
                OwnPred = (crop_model1.predict(image)).astype(np.float64)
                mask = torch.Tensor(OwnPred)
                mask = mask.squeeze(1)
                mask[mask > 0.35] = 1.0
                mask[mask <= 0.35] = 0.0

                # We get the unique colors, as these would be the object ids.
                obj_ids = torch.unique(mask)

                # first id is the background, so remove it.
                obj_ids = obj_ids[1:]

                # split the color-encoded mask into a set of boolean masks.
                # Note that this snippet would work as well if the masks were float values instead of ints.

                masks = mask == obj_ids[:, None, None]
                box = masks_to_boxes(masks)
                x1 = max(0, box[0][0] - 30)
                x2 = min(255, box[0][2] + 30)
                y1 = max(0, box[0][1] - 30)
                y2 = min(255, box[0][3] + 30)
                image = image.transpose((0, 2, 3, 1))

                image = image[0]
                fy = h/256
                fx = w/256
                # im = im.astype(np.float64) * 255.0
                cropped_im = image[int(
                    y1*fx):int(y2*fx), int(x1*fy):int(x2*fy), :]

                cropped_im = transform(cropped_im)
                cropped_images.append(cropped_im)
                cropped_im = cropped_im * 255
                ###################################
                # save_image = np.array(cropped_im)
                # # save_image = save_image.transpose((1, 2, 0))
                # # Assuming 'cropped_im' is your numpy array with shape (512, 512, 3) or (3, 512, 512)
                # # Ensure the values are in the range [0, 255]
                # save_image = save_image.astype(np.uint8)

                # # If the array shape is (3, 512, 512), transpose it to (512, 512, 3)
                # if save_image.shape[0] == 3:
                #     save_image = np.transpose(save_image, (1, 2, 0))
                # save_image = Image.fromarray(save_image)

                # # Save the image to a file (e.g., in PNG format)
                # save_image.save(f"output_image_{index}.png")
            except:
                try:
                    OwnPred = (crop_model2.predict(image)).astype(np.float64)
                    mask = torch.Tensor(OwnPred)
                    mask = mask.squeeze(1)
                    mask[mask > 0.35] = 1.0
                    mask[mask <= 0.35] = 0.0

                    # We get the unique colors, as these would be the object ids.
                    obj_ids = torch.unique(mask)

                    # first id is the background, so remove it.
                    obj_ids = obj_ids[1:]
                    # split the color-encoded mask into a set of boolean masks.
                    # Note that this snippet would work as well if the masks were float values instead of ints.
                    masks = mask == obj_ids[:, None, None]
                    box = masks_to_boxes(masks)
                    # pad_x = (box[0][2] - box[0][0]) * 0.3
                    # pad_y = (box[0][3] - box[0][1]) * 0.3

                    # pad = max(pad_x, pad_y)
                    # pad = max(pad, 20)
                    # transform = transforms.Compose([
                    #     transforms.ToTensor(),
                    #     transforms.Resize((512, 512))
                    # ])
                    x1 = max(0, box[0][0] - 30)
                    x2 = min(255, box[0][2] + 30)
                    y1 = max(0, box[0][1] - 30)
                    y2 = min(255, box[0][3] + 30)
                    image = image.transpose((0, 2, 3, 1))

                    image = image[0]
                    fy = h/256
                    fx = w/256
                    # im = im.astype(np.float64) * 255.0
                    cropped_im = image[int(
                        y1*fx):int(y2*fx), int(x1*fy):int(x2*fy), :]
                    cropped_im = transform(cropped_im)
                    cropped_images.append(cropped_im)
                    cropped_im = cropped_im * 255
                    ####################################
                    # save_image = np.array(cropped_im)
                    # # save_image = save_image.transpose((1, 2, 0))
                    # # Assuming 'cropped_im' is your numpy array with shape (512, 512, 3) or (3, 512, 512)
                    # # Ensure the values are in the range [0, 255]
                    # save_image = save_image.astype(np.uint8)

                    # # If the array shape is (3, 512, 512), transpose it to (512, 512, 3)
                    # if save_image.shape[0] == 3:
                    #     save_image = np.transpose(save_image, (1, 2, 0))
                    # save_image = Image.fromarray(save_image)

                    # # Save the image to a file (e.g., in PNG format)
                    # save_image.save(f"output_image_{index}.png")
                except:
                    try:
                        OwnPred = (crop_model3.predict(
                            image_128)).astype(np.float64)
                        mask = torch.Tensor(OwnPred)
                        mask = mask.squeeze(1)
                        mask[mask > 0.35] = 1.0
                        mask[mask <= 0.35] = 0.0

                        # We get the unique colors, as these would be the object ids.
                        obj_ids = torch.unique(mask)

                        # first id is the background, so remove it.
                        obj_ids = obj_ids[1:]

                        # split the color-encoded mask into a set of boolean masks.
                        # Note that this snippet would work as well if the masks were float values instead of ints.
                        masks = mask == obj_ids[:, None, None]
                        box = masks_to_boxes(masks)

                        # pad_x = (box[0][2] - box[0][0]) * 0.3
                        # pad_y = (box[0][3] - box[0][1]) * 0.3

                        # pad = max(pad_x, pad_y)
                        # pad = max(pad, 20)
                        # x1 = max(0, box[0][0] - pad)
                        # x2 = min(255, box[0][2] + pad)
                        # y1 = max(0, box[0][1] - pad)
                        # y2 = min(255, box[0][3] + pad)

                        x1 = max(0, box[0][0] - 30)
                        x2 = min(255, box[0][2] + 30)
                        y1 = max(0, box[0][1] - 30)
                        y2 = min(255, box[0][3] + 30)
                        image_128 = image_128.transpose((0, 2, 3, 1))

                        image_128 = image_128[0]
                        fy = h/128
                        fx = w/128
                        # im = im.astype(np.float64) * 255.0
                        cropped_im = image_128[int(
                            y1*fx):int(y2*fx), int(x1*fy):int(x2*fy), :]
                        cropped_im = cropped_im * 255
                        cropped_im = transform(cropped_im)
                        cropped_images.append(cropped_im)
                        ##################################
                        # save_image = np.array(cropped_im)
                        # # save_image = save_image.transpose((1, 2, 0))
                        # # Assuming 'cropped_im' is your numpy array with shape (512, 512, 3) or (3, 512, 512)
                        # # Ensure the values are in the range [0, 255]
                        # save_image = save_image.astype(np.uint8)

                        # # If the array shape is (3, 512, 512), transpose it to (512, 512, 3)
                        # if save_image.shape[0] == 3:
                        #     save_image = np.transpose(
                        #         save_image, (1, 2, 0))
                        # save_image = Image.fromarray(save_image)

                        # # Save the image to a file (e.g., in PNG format)
                        # save_image.save(f"output_image_{index}.png")
                    except:
                        try:
                            OwnPred = (crop_model4.predict(
                                image_128)).astype(np.float64)
                            mask = torch.Tensor(OwnPred)
                            mask = mask.squeeze(1)
                            mask[mask > 0.35] = 1.0
                            mask[mask <= 0.35] = 0.0

                            # We get the unique colors, as these would be the object ids.
                            obj_ids = torch.unique(mask)

                            # first id is the background, so remove it.
                            obj_ids = obj_ids[1:]

                            # split the color-encoded mask into a set of boolean masks.
                            # Note that this snippet would work as well if the masks were float values instead of ints.
                            masks = mask == obj_ids[:, None, None]
                            box = masks_to_boxes(masks)
                            print(box.shape)
                            print(box)

                            # pad_x = (box[0][2] - box[0][0]) * 0.3
                            # pad_y = (box[0][3] - box[0][1]) * 0.3

                            # pad = max(pad_x, pad_y)
                            # pad = max(pad, 20)

                            # pad_x = (box[0][2] - box[0][0]) * 0.3
                            # pad_y = (box[0][3] - box[0][1]) * 0.3

                            # pad = max(pad_x, pad_y)
                            # pad = max(pad, 20)
                            # x1 = max(0, box[0][0] - pad)
                            # x2 = min(255, box[0][2] + pad)
                            # y1 = max(0, box[0][1] - pad)
                            # y2 = min(255, box[0][3] + pad)

                            x1 = max(0, box[0][0] - 30)
                            x2 = min(255, box[0][2] + 30)
                            y1 = max(0, box[0][1] - 30)
                            y2 = min(255, box[0][3] + 30)
                            image_128 = image_128.transpose(
                                (0, 2, 3, 1))

                            image_128 = image_128[0]
                            fy = h/128
                            fx = w/128
                            # im = im.astype(np.float64) * 255.0
                            cropped_im = image_128[int(
                                y1*fx):int(y2*fx), int(x1*fy):int(x2*fy), :]
                            cropped_im = cropped_im * 255
                            cropped_im = transform(cropped_im)
                            cropped_images.append(cropped_im)
                            ###################################
                            # save_image = np.array(cropped_im)
                            # # save_image = save_image.transpose((1, 2, 0))
                            # # Assuming 'cropped_im' is your numpy array with shape (512, 512, 3) or (3, 512, 512)
                            # # Ensure the values are in the range [0, 255]
                            # save_image = save_image.astype(np.uint8)

                            # # If the array shape is (3, 512, 512), transpose it to (512, 512, 3)
                            # if save_image.shape[0] == 3:
                            #     save_image = np.transpose(
                            #         save_image, (1, 2, 0))
                            # save_image = Image.fromarray(save_image)

                            # Save the image to a file (e.g., in PNG format)
                            # save_image.save(
                            #     f"output_image_{index}.png")
                        except:
                            image = image.transpose((0, 2, 3, 1))
                            image = image[0]
                            image = transform(image)
                            cropped_images.append(image)
        cropped_im = torch.stack(cropped_images, dim=0)
        return cropped_im


def get_unet_light(img_rows=256, img_cols=256):
    inputs = Input((3, img_rows, img_cols))
    conv1 = Conv2D(32, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv2D(32, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D(pool_size=(
        2, 2), data_format='channels_first')(conv1)

    conv2 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D(pool_size=(
        2, 2), data_format='channels_first')(conv2)

    conv3 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv3)
    pool3 = MaxPooling2D(pool_size=(
        2, 2), data_format='channels_first')(conv3)

    conv4 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv4)
    pool4 = MaxPooling2D(pool_size=(
        2, 2), data_format='channels_first')(conv4)

    conv5 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv5)

    up6 = Concatenate(axis=1)(
        [UpSampling2D(size=(2, 2), data_format='channels_first')(conv5), conv4])
    conv6 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(up6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv6)

    up7 = Concatenate(axis=1)(
        [UpSampling2D(size=(2, 2), data_format='channels_first')(conv6), conv3])
    conv7 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(up7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv7)

    up8 = Concatenate(axis=1)(
        [UpSampling2D(size=(2, 2), data_format='channels_first')(conv7), conv2])
    conv8 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(up8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(64, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv8)

    up9 = Concatenate(axis=1)(
        [UpSampling2D(size=(2, 2), data_format='channels_first')(conv8), conv1])
    conv9 = Conv2D(32, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(up9)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(32, kernel_size=3, activation='relu',
                   padding='same', data_format='channels_first')(conv9)

    conv10 = Conv2D(1, kernel_size=1, activation='sigmoid',
                    padding='same', data_format='channels_first')(conv9)
    # conv10 = Flatten()(conv10)

    model = Model(inputs=inputs, outputs=conv10)

    return model


# model1_256 = get_unet_light(img_rows=256, img_cols=256)
# model1_256.load_weights('rim_256.hdf5')
# model2_256 = get_unet_light(img_rows=256, img_cols=256)
# model2_256.load_weights('drions.hdf5')
# model3_128 = get_unet_light(img_rows=128, img_cols=128)
# model3_128.load_weights('rim_128.hdf5')
# model4_128 = get_unet_light(img_rows=128, img_cols=128)
# model4_128.load_weights('drions.hdf5')

def modify_dataframe(dataframe, ratio=0.1):
    nrg_index = dataframe.index[dataframe['Final Label'] == 'NRG'].tolist(
    )

    rg_index = dataframe.index[dataframe['Final Label'] == 'RG'].tolist(
    )
    count_rg = len(rg_index)
    count_nrg = len(nrg_index)

    desired_nrg_count = int(ratio * count_nrg)
    desired_nrg_index = nrg_index[:desired_nrg_count]
    new_dataframe = dataframe.loc[desired_nrg_index].copy()

    # You might want to return the modified DataFrame if needed
    return new_dataframe
