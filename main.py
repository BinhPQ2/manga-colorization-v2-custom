import subprocess

inference_script = "manga-colorization-v2-custom/inference_v2.py"
# inference_script = "inference.py" # need to cd to the directory containing this script
image_path = r"D:\Download\Tu_Lieu\Cao_Hoc\Ky_3\Multimedia\FinalExam\manga_read_along\magi_functional\data_test\personal_data\Ruri_Dragon\raw"
denoiser_sigma = 0
use_gpu = True
denoiser_path = r"D:\Download\Tu_Lieu\Cao_Hoc\Ky_3\Multimedia\FinalExam\manga_read_along\manga-colorization-v2-custom\denoising\models"
generator_path = r"D:\Download\Tu_Lieu\Cao_Hoc\Ky_3\Multimedia\FinalExam\manga_read_along\manga-colorization-v2-custom\networks\generator.zip"

subprocess.run(['python', inference_script, '-p', image_path, '-des_path', denoiser_path, "-gen", generator_path,'-ds', str(denoiser_sigma)] + (['--gpu'] if use_gpu else []))
