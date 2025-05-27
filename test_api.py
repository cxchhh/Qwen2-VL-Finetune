import mediapy as media
import numpy as np

video_path = '000000_vid.mp4'  # 替换为你的视频路径
num_frames=15
# 读取整个视频，返回一个形状为 [num_frames, height, width, 3] 的 NumPy 数组
frames = media.read_video(video_path)


fps = frames.metadata.fps

print(f'视频读取成功！fps:{fps}, 形状: {frames.shape}')




# media.write_video("fat_"+video_path, np.repeat(frames, repeats=2, axis=0), fps=fps)

# output_path1 = "extract1.mp4"
# output_path2 = "extract2.mp4"
# output_path3 = "extract3.mp4"
# # 保存为新视频（确保使用原始 fps）
# media.write_video(output_path1, np.array(frames[:15]), fps=fps)
# media.write_video(output_path2, np.array(frames[15:28]), fps=fps)
# media.write_video(output_path3, np.array(frames[28:]), fps=fps)



# It's a 64-frame video that contains a robotic task "turn of the light". Please cut it into 2-5 clips, marked with precise init frame number, with each representing a single movement in: move upward/downward/left/right/to <target object>, push upward/downward, open/close gripper. write it in json format {'init frame': <frame number>, 'movement': "<movement>"}