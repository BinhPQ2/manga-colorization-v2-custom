import subprocess

# inference_script = "inference_v2.py" # need to cd to the directory containing this script
inference_script = "inference.py" # need to cd to the directory containing this script
image_path = r"D:\Download\Tu_Lieu\Cao_Hoc\Ky_3\Multimedia\FinalExam\manga_read_along\magi_functional\data_test\personal_data\Ruri_Dragon\raw"
denoiser_sigma = 0
use_gpu = True

subprocess.run(['python', inference_script, '-p', image_path, '-ds', str(denoiser_sigma)] + (['--gpu'] if use_gpu else []))
