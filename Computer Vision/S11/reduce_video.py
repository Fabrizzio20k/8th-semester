import ffmpeg

input_path = "a.mp4"
output_path = "video_reducido.mp4"

(
    ffmpeg
    .input(input_path)
    .filter('fps', fps=1)
    .filter('scale', 1280, 720)
    .output(output_path, vcodec='libx264', crf=23, preset='fast')
    .overwrite_output()
    .run()
)
