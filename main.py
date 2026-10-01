import argparse
import sys

import cv2
import numpy as np


_CORNER_CFG = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7,
)

_LK_CFG = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

_PALETTE = np.random.randint(0, 255, (100, 3))


class FlowTracker:
    def __init__(self, source: str):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"ERROR: cannot open '{source}'", file=sys.stderr)
            sys.exit(1)

        ok, seed_frame = self.cap.read()
        if not ok:
            print("ERROR: cannot read first frame", file=sys.stderr)
            sys.exit(1)

        self.prev = cv2.cvtColor(seed_frame, cv2.COLOR_BGR2GRAY)
        self.anchors = cv2.goodFeaturesToTrack(self.prev, mask=None, **_CORNER_CFG)
        self.trails = np.zeros_like(seed_frame)

    def _refresh_anchors(self, gray, frame):
        self.prev = gray.copy()
        self.anchors = cv2.goodFeaturesToTrack(gray, mask=None, **_CORNER_CFG)
        self.trails = np.zeros_like(frame)

    def _step(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        moved, flags, _ = cv2.calcOpticalFlowPyrLK(
            self.prev, gray, self.anchors, None, **_LK_CFG
        )

        survived = flags.ravel() == 1
        src = self.anchors[survived]
        dst = moved[survived]

        if len(dst) < 5:
            self._refresh_anchors(gray, frame)
            return None

        for idx, (a, b) in enumerate(zip(src, dst)):
            color = _PALETTE[idx % 100].tolist()
            x0, y0 = a.ravel().astype(int)
            x1, y1 = b.ravel().astype(int)
            cv2.line(self.trails, (x0, y0), (x1, y1), color, 2)
            cv2.circle(frame, (x1, y1), 4, color, -1)

        canvas = cv2.add(frame, self.trails)
        cv2.putText(
            canvas,
            f"tracking: {len(dst)} pts",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        self.prev = gray.copy()
        self.anchors = dst.reshape(-1, 1, 2)
        return canvas

    def run(self):
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break

                result = self._step(frame)
                if result is None:
                    continue

                cv2.imshow("optical flow", result)
                if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    FlowTracker(ap.parse_args().video).run()


if __name__ == "__main__":
    main()
