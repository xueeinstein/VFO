import cv2


class MovieWriter(object):
    def __init__(self, file_name, frame_size, fps):
        """
        frame_size is (w, h)
        """
        self._frame_size = frame_size
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.vout = cv2.VideoWriter()
        success = self.vout.open(file_name, fourcc, fps, frame_size, True)
        if not success:
            print("Create movie failed: {0}".format(file_name))

    def add_frame(self, frame):
        """
        frame shape is (h, w, 3), dtype is np.uint8
        """
        # TODO: update this hard code for Atari NoFrameskip-v4 envs
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame[:, :, -1:], cv2.COLOR_GRAY2BGR)
        self.vout.write(frame)

    def close(self):
        self.vout.release()
        self.vout = None
