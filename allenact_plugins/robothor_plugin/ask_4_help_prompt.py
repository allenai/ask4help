# Copyright Allen Institute for Artificial Intelligence 2017
"""
ai2thor.local_actions

Action runner and action list for step actions that are intercepted
to run locally and be resolved in python without going through Unity

"""
import cv2
import copy
from ai2thor.interact import DefaultActions
from ai2thor.interact_navigation import NavigationPrompt
import numpy as np 

class Ask4HelpActionRunner(object):
    def __init__(
            self,
            enabled_actions,
            scale=1.0
    ):
        self.interactive_prompt = NavigationPrompt(
            default_actions=[DefaultActions[a] for a in enabled_actions]
        )
        self.target_object_type = ""
        self.scale = scale

    def ObjectNavHumanAction(self, action, controller):
        img = controller.last_event.cv2img[:, :, :]
        # dst = cv2.resize(
        #     img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4
        # )

        print("Select next action")

        # Simpler version instead of interact controller
        # actions = {"1": "RotateLeft", "2": "RotateRight"}
        # for key, a in actions.items():
        #     print("({}) {}".format(key, a))
        #
        # cv2.namedWindow("image")
        # cv2.imshow("image", img)
        # cv2.waitKey(1)

        # choice = str(input())
        # result = ""
        # if choice in actions:
        #     result = actions[choice]
        # else:
        #     raise ValueError("Invalid choice `{}`, please choose a number from `{}`".format(choice, actions.keys()))
        #

        cv2.namedWindow("image")
        cv2.setWindowProperty("image", cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow("image", 0, 0)
        resized_main = self._resize_image(img, int(controller.width * self.scale), int(controller.height * self.scale))
        cv2.imshow("image", resized_main)

        print("--------- 3rd party camera ")
        print(len(controller.last_event.third_party_camera_frames))

        if len(controller.last_event.third_party_camera_frames) > 0 and len(
                controller.last_event.third_party_camera_frames[0]):
            # [...,::-1]

            print("instance mask")

            id = [x['objectId'] for x in controller.last_event.metadata["objects"] if x['objectType'] == self.target_object_type]
            
            print(id)

            instance_detect_dict = controller.last_event.object_id_to_color

            # print (instance_detect_dict.keys())
            if id[0] in instance_detect_dict:
                bbox = instance_detect_dict[id[0]]
                print (bbox)
            else:
                print ('----- didnt find it')
            # exit()
            print(controller.last_event.object_id_to_color.keys())
            if id[0] in controller.last_event.object_id_to_color:
                print("id {} in object_id_to_color".format(id[0]))
            else:
                raise Exception("id {} not object_id_to_color".format(id[0]))

            target_segmentation_color = controller.last_event.object_id_to_color[id[0]]
            print("------ id {} color {}".format(id[0], controller.last_event.object_id_to_color[id[0]]))
            # print(controller.last_event.object_id_to_color[id[0]])

            resized = controller.last_event.third_party_camera_frames[0][..., ::-1][:, :, :]

            if len(controller.last_event.third_party_instance_segmentation_frames) > 0:
                im_og = controller.last_event.third_party_instance_segmentation_frames[0]
                im3 = controller.last_event.third_party_instance_segmentation_frames[0][..., ::-1][:, :, :]

                # Y,X = np.where(np.all(im_og==bbox,axis=2))
                # if np.all(im_og==bbox,axis=2).sum()==0:
                #     print ('missing')
                # x1,x2 = min(X),max(X)
                # y1,y2 = min(Y),max(Y)

            im = controller.last_event.third_party_camera_frames[0][..., ::-1][:, :, :]

            # cv2.namedWindow("top_down", cv2.WINDOW_AUTOSIZE)
            # cv2.setWindowProperty("top_down", cv2.WND_PROP_TOPMOST, 1)
            # cv2.imshow("top_down", resized)

        print("Segmentation available")
        print(controller.last_event.instance_segmentation_frame is not None)
        # if  controller.last_event.instance_segmentation_frame is not None:
        #     im2 =  controller.last_event.instance_segmentation_frame[...,::-1][:, :, :]
        #     cv2.namedWindow("seg")
        #     cv2.setWindowProperty("seg", cv2.WND_PROP_TOPMOST, 1)
        #     cv2.imshow("seg", im2)

        # third_party_instance_segmentation_frames

        print("third party seg")
        print(len(controller.last_event.third_party_instance_segmentation_frames))

        if len(controller.last_event.third_party_instance_segmentation_frames) > 0:
            im_og = controller.last_event.third_party_instance_segmentation_frames[0]
            im3 = controller.last_event.third_party_instance_segmentation_frames[0][..., ::-1][:, :, :]
            # cv2.namedWindow("seg2")
            # cv2.setWindowProperty("seg2", cv2.WND_PROP_TOPMOST, 1)
            # cv2.imshow("seg2", im3)

            np.array(im3)

            color = np.asarray(target_segmentation_color[::-1])

            # x1,x2 = min(X),max(X)
            # y1,y2 = min(Y),max(Y)

            print("shape {}, color {}".format(np.shape(im3), color))
            indices_y, indices_x =np.where(np.all(im3 == np.asarray(color), axis=2))

            print("--- indices")
            print(indices_y)
            print(indices_x)

            print("sample color")
            print(im3[0, 0])

            x1, x2 = min(indices_x), max(indices_x)
            y1, y2 = min(indices_y), max(indices_y)

            cv2.namedWindow("top_down", cv2.WINDOW_AUTOSIZE)
            resized = cv2.rectangle(cv2.UMat(resized),(x1,y1),(x2,y2),(0,0,255),-1)
            # cv2.namedWindow("seg2")
            # cv2.setWindowProperty("seg2", cv2.WND_PROP_TOPMOST, 1)
            # cv2.imshow("seg2", im3)
            cv2.moveWindow("top_down", int(controller.width * self.scale), 0)
            cv2.setWindowProperty("top_down", cv2.WND_PROP_TOPMOST, 1)

            # cv2.imshow("top_down", resized)
            resized = self._resize_image(resized, int(controller.width * self.scale), int(controller.height * self.scale))
            cv2.imshow("top_down", resized)



            # Y,X

            # Y,X = np.where(np.all(im_og==bbox,axis=2))
            # if np.all(im_og==bbox,axis=2).sum()==0:

        # if len(controller.last_event.third_party_instance_segmentation_frames) > 0:
        #     im_og = controller.last_event.third_party_instance_segmentation_frames[0]
        #     im3 = controller.last_event.third_party_instance_segmentation_frames[0][..., ::-1][:, :, :]
            # print (im3.shape,'segmentation shape')
            # print (np.all(im3==bbox,axis=2).sum())
            # print (np.all(im_og==bbox,axis=2).sum())
            # Y,X = np.where(np.all(im_og==bbox,axis=2))
            # if np.all(im_og==bbox,axis=2).sum()==0:
            #     print ('missing')
            # x1,x2 = min(X),max(X)
            # y1,y2 = min(Y),max(Y)
            # cv2.rectangle(cv2.UMat(im3),(x1,y1),(x2,y2),(255,0,0),-1)
            # cv2.namedWindow("seg2")
            # cv2.setWindowProperty("seg2", cv2.WND_PROP_TOPMOST, 1)
            # cv2.imshow("seg2", im3)

        # TODO  perhaps opencv not needed, just a good resolution for THOR
        cv2.waitKey(1)

        # TODO modify interactive controller accordingly, or use simple version commented above
        result = self.interactive_prompt.interact(
            controller,
            step=False
        )

        # Deepcopy possibly not necessary
        event_copy = copy.deepcopy(controller.last_event)
        event_copy.metadata["actionReturn"] = result
        return event_copy

    def _resize_image(self, img, width, height):
        return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
