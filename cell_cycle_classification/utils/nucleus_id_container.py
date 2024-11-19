from typing import Optional


def get_track_id(track_spots, nucleus_id):
    for track_id, spots in track_spots.items():
        if nucleus_id in spots:
            return track_id
    return None


class NucleusIdContainer:
    """
    3 digits for track id
    3 digits for frame
    5 digits for spot id (sometimes 6...)
    """

    def __init__(self, nucleus_id: Optional[str] = None) -> None:

        self.track_id: Optional[int] = None
        self.frame: Optional[int] = None
        self.spot_id: Optional[int] = None
        self.video_id: Optional[int] = None
        self.phase: int = -1

        if nucleus_id is not None:
            if "_n" in nucleus_id:
                video_id, nucleus_id = nucleus_id.split("_n")
                self.video_id = int(
                    "".join([char for char in video_id if char.isdigit()])
                )

            nucleus_id = nucleus_id.split(".")[0]

            # Some have "_c", some not
            if "_c" in nucleus_id:
                nucleus_id, phase = nucleus_id.split("_c")[:2]
                self.phase = int(phase)
            else:
                self.phase = -1

            # NB: should be 11, be maybe some nucleus id are longer than 6 digits
            if len(nucleus_id) in [11, 12]:
                self.track_id = int(nucleus_id[:3])
                self.frame = int(nucleus_id[3:6])
                self.spot_id = int(nucleus_id[6:])
            else:  # other cases
                self.track_id = 0
                self.frame = 0
                self.spot_id = int(nucleus_id)

    def init_from_spot(self, spot, track_spots):
        track_id = get_track_id(track_spots, int(spot["@ID"]))
        if track_id is None:
            return

        self.track_id = int(track_id)
        self.frame = int(spot["@FRAME"])
        self.spot_id = int(spot["@ID"])

    def get_id_str(self) -> str:
        assert self.track_id is not None
        assert self.frame is not None
        assert self.spot_id is not None

        track_id = str(self.track_id).rjust(3, "0")
        frame = str(self.frame).rjust(3, "0")
        spot_id = str(self.spot_id).rjust(5, "0")

        return track_id + frame + spot_id

    def get_video_track_id(self):
        assert self.video_id is not None
        assert self.track_id is not None

        video_id = str(self.video_id).rjust(6, "0")  # r**c**f**
        track_id = str(self.track_id).rjust(3, "0")

        return int(video_id + track_id)
