

class LandmarkMap:
    def __init__(self, landmarks):
        self.landmarks = {int(l[0]): tuple(l[1:]) for l in landmarks}

    def is_added(self, lid):
        return lid in self.landmarks.keys()

    def add(self, landmark):
        lid, lx, ly = landmark

        if not self.is_added(lid):
            self.landmarks[lid] = (lx, ly)

    def get(self, lid):
        if self.is_added(lid):
            return self.landmarks[lid]