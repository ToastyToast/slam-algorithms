

def read_simple_data(filename):
    odometry = []
    sensor = []

    with open(filename, 'r') as f:
        meas_pack = []
        for line in f:
            data = line.strip().split(" ")

            if data[0] == "ODOMETRY":
                if len(meas_pack) != 0:
                    sensor.append(meas_pack)
                    meas_pack = []
                # rotation1, translation, rotation2
                data[1:] = [float(d) for d in data[1:]]
                odometry.append(tuple(data[1:]))
            elif data[0] == "SENSOR":
                # id of observed landmark
                data[1] = int(data[1])
                # range, bearing
                data[2:] = [float(d) for d in data[2:]]

                meas_pack.append(tuple(data[1:]))

    return odometry, sensor


def read_simple_world(filename):
    landmarks = []

    with open(filename, 'r') as f:
        for line in f:
            data = line.strip().split(" ")
            data[0] = int(data[0])
            data[1:] = [float(d) for d in data[1:]]

            landmarks.append(tuple(data))

    return landmarks