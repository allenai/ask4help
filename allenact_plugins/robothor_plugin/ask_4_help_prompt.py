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


class Ask4HelpActionRunner(object):
    def __init__(
            self,
            enabled_actions
    ):
        self.interactive_prompt = NavigationPrompt(
            default_actions=[DefaultActions[a] for a in enabled_actions]
        )
        self.target_object_type = ""

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
        cv2.imshow("image", img)

        print("--------- 3rd party camera ")
        print(len(controller.last_event.third_party_camera_frames))

        if len(controller.last_event.third_party_camera_frames) > 0 and len(
                controller.last_event.third_party_camera_frames[0]):
            # [...,::-1]

            print("instance mask")

            id = [x['objectId'] for x in controller.last_event.metadata["objects"] if x['objectType'] == self.target_object_type]

            print(id)
            print(controller.last_event.object_id_to_color[id[0]])

            # im = Image.fromarray(controller.last_event.third_party_camera_frames[0][:, :, :])
            im = controller.last_event.third_party_camera_frames[0][..., ::-1][:, :, :]
            cv2.namedWindow("top_down")
            cv2.setWindowProperty("top_down", cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow("top_down", im)

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
            im3 = controller.last_event.third_party_instance_segmentation_frames[0][..., ::-1][:, :, :]
            cv2.namedWindow("seg2")
            cv2.setWindowProperty("seg2", cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow("seg2", im3)

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

