import pandas as pd
import os
import sys

column_list = ["time step", "agent_x", "agent_y", "rel_x0", "rel_y0", "rel_x1", "rel_y1", "rel_x2", "rel_y2",
               "rel_x3", "rel_y3", "rel_x4", "rel_y4", "rel_x5", "rel_y5", "rel_x6", "rel_y6", "rel_x7", "rel_y7",
               "rel_x8", "rel_y8", "rel_x9", "rel_y9"]
num_rowsX = 10
num_agents = 10

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("usage: specify input folder and output name as arguments.")

    compiled_df = pd.DataFrame(columns=column_list)
    for filename in os.scandir(sys.argv[1]):
        print("Processing {}".format(filename))

        df = pd.read_csv(os.path.join(filename))
        new_df = pd.DataFrame(columns=column_list)
        new_df["time step"] = df["time step"]

        # get agent coords
        for label, content in df.items():
            if "role" in label and "agent" in content[0]:
                agent_col_idx = df.columns.get_loc(label)
                new_df["agent_x"] = df.iloc[:, agent_col_idx + 2]
                new_df["agent_y"] = df.iloc[:, agent_col_idx + 3]
                break

        # get relative coords
        for i in range(num_agents):
            x_col = df[" x" + str(i)]
            y_col = df[" y" + str(i)]
            if x_col[num_rowsX - 1] == 0:
                new_df["rel_x" + str(i)] = 0
                new_df["rel_y" + str(i)] = 0
            else:
                new_df["rel_x" + str(i)] = df[" x" + str(i)] - new_df["agent_x"]
                new_df["rel_y" + str(i)] = df[" y" + str(i)] - new_df["agent_y"]

        compiled_df = compiled_df.append(new_df, ignore_index=True)

    pd.DataFrame.to_csv(compiled_df, os.path.join('..', 'data', sys.argv[2]), index=False)

