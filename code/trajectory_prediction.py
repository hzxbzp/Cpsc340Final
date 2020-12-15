import numpy as np
import pandas as pd
import sys

num_actors = 10

class Actor:
    def __init__(self, pos, vel, acc, present):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.present = present
        self.stopping = False
        self.resuming = False
        self.target_vel = None

    def __str__(self):
        return "actor: " + str((str(self.pos), str(self.vel), str(self.acc)))

    def __repr__(self):
        return "actor: " + str((str(self.pos), str(self.vel), str(self.acc)))

class TrajectoryPrediction:

    def __init__(self, data, collision_radius=1, stopping_time=5, resuming_time=4):
        self.actors = []
        self.collision_radius = collision_radius
        self.stopping_time = stopping_time
        self.resuming_time = resuming_time
        self.columns = ["time step", "x0", "y0", "x1", "y1", "x2", "y2",
                        "x3", "y3", "x4", "y4", "x5", "y5", "x6", "y6",
                        "x7", "y7", "x8", "y8", "x9", "y9"]
        final_rows = data[-4:]
        for i in range(num_actors):
            present = all(final_rows[" present" + str(i)])
            pos_x_col = final_rows[" x" + str(i)]
            pos_y_col = final_rows[" y" + str(i)]
            pos = np.array([pos_x_col[10], pos_y_col[10]])
            v_init = np.array([pos_x_col[8] - pos_x_col[7], pos_y_col[8] - pos_y_col[7]])
            v_final = np.array([pos_x_col[10] - pos_x_col[9], pos_y_col[10] - pos_y_col[9]])
            acc = (v_final - v_init) / 3
            if not present:
                v_final = np.array([0, 0])
                acc = np.array([0, 0])
            actor = Actor(pos, v_final, acc, present)
            self.actors.append(actor)

    def predict(self, steps=30):
        df = pd.DataFrame(columns=self.columns)
        for step in range(1, steps+1):
            data_row = [step*100]
            # Check for collisions
            colliding_actors = set()
            for i in range(len(self.actors)):
                a1 = self.actors[i]
                if not a1.present:
                    continue
                for j in range(i+1, len(self.actors)):
                    a2 = self.actors[j]
                    if not a2.present:
                        continue
                    if np.linalg.norm(a1.pos - a2.pos) < self.collision_radius:
                        colliding_actors.add(a1)
                        colliding_actors.add(a2)

            for actor in self.actors:
                if actor in colliding_actors:
                    if not actor.stopping:
                        # start actor decelerating
                        actor.target_vel = actor.vel
                        decel_factor = -np.linalg.norm(actor.vel) / self.stopping_time
                        actor.acc = actor.vel * decel_factor
                    elif np.linalg.norm(actor.vel) <= 0:
                        # actor has stopped, cease decelerating
                        actor.vel = np.array([0, 0])
                        actor.acc = np.array([0, 0])
                elif actor.stopping:
                    # actor is no longer in collision radius, start moving again
                    actor.stopping = False
                    actor.resuming = True
                    accel_factor = np.linalg.norm(actor.target_vel) / self.resuming_time
                    actor.acc = actor.target_vel * accel_factor
                elif actor.resuming and np.linalg.norm(actor.vel) >= np.linalg.norm(actor.target_vel):
                    # actor has reached target speed, stop accelerating
                    actor.resuming = False
                    actor.vel = actor.target_vel
                    actor.acc = np.array([0, 0])

                actor.pos += actor.vel
                actor.vel += actor.acc
                data_row.extend(actor.pos)

            df = df.append(pd.Series(data_row, index=df.columns), ignore_index=True)

        return df


if __name__ == "__main__":
    dataframe = pd.read_csv(sys.argv[1])

    predictor = TrajectoryPrediction(dataframe)
    print(predictor.predict())
