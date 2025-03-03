'''process_assets.py'''
# 预处理图片和视频，建立索引，加快搜索速度
import concurrent.futures
import logging
import traceback

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from tqdm import trange
from transformers import CLIPModel, CLIPProcessor

from config import *

import subprocess
import whisper
from moviepy import VideoFileClip

logger = logging.getLogger(__name__)

logger.info("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(torch.device(DEVICE))
clip_model.eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
logger.info("CLIP model loaded.")

whisper_model = whisper.load_model("base")

def get_image_feature(images):
    """
    :param images: 图片
    :return: feature
    """
    feature = None
    try:
        inputs = clip_processor(images=images, return_tensors="pt").to(torch.device(DEVICE))
        feature = clip_model.get_image_features(inputs['pixel_values']).detach().cpu().numpy()
    except Exception as e:
        logger.warning(f"处理图片报错：{repr(e)}")
        traceback.print_stack()
    return feature


def get_image_data(path: str, ignore_small_images: bool = True):
    """
    获取图片像素数据，如果出错返回 None
    :param path: string, 图片路径
    :param ignore_small_images: bool, 是否忽略尺寸过小的图片
    :return: <class 'numpy.nparray'>, 图片数据，如果出错返回 None
    """
    try:
        image = Image.open(path)
        if ignore_small_images:
            width, height = image.size
            if width < IMAGE_MIN_WIDTH or height < IMAGE_MIN_HEIGHT:
                return None
                # processor 中也会这样预处理 Image
        # 在这里提前转为 np.array 避免到时候抛出异常
        image = image.convert('RGB')
        image = np.array(image)
        return image
    except Exception as e:
        logger.warning(f"打开图片报错：{path} {repr(e)}")
        return None


def process_image(path, ignore_small_images=True):
    """
    处理图片，返回图片特征
    :param path: string, 图片路径
    :param ignore_small_images: bool, 是否忽略尺寸过小的图片
    :return: <class 'numpy.nparray'>, 图片特征
    """
    image = get_image_data(path, ignore_small_images)
    if image is None:
        return None
    feature = get_image_feature(image)
    return feature


def process_images(path_list, ignore_small_images=True):
    """
    处理图片，返回图片特征
    :param path_list: string, 图片路径列表
    :param ignore_small_images: bool, 是否忽略尺寸过小的图片
    :return: <class 'numpy.nparray'>, 图片特征
    """
    images = []
    for path in path_list.copy():
        image = get_image_data(path, ignore_small_images)
        if image is None:
            path_list.remove(path)
            continue
        images.append(image)
    if not images:
        return None, None
    feature = get_image_feature(images)
    return path_list, feature


def process_web_image(url):
    """
    处理网络图片，返回图片特征
    :param url: string, 图片URL
    :return: <class 'numpy.nparray'>, 图片特征
    """
    try:
        image = Image.open(requests.get(url, stream=True).raw)
    except Exception as e:
        logger.warning("获取图片报错：%s %s" % (url, repr(e)))
        return None
    feature = get_image_feature(image)
    return feature

def extract_audio_from_video(video_path, audio_path):
    # command = [
    #     "ffmpeg", "-i", video_path, 
    #     "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
    #     audio_path
    # ]
    # subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if(os.path.exists(audio_path)):#如果文件已经存在，则不再提取
        return
    video = VideoFileClip(video_path)
    audio_clip = video.audio
    audio_clip.write_audiofile(audio_path, codec='pcm_s16le')


def get_audio_transcript_for_time(current_time, audio_segments):
    """
    获取当前时间对应的转录文本
    :param current_time: 当前视频帧的时间戳
    :param audio_segments: Whisper模型的转录结果片段
    :return: 对应时间戳的转录文本
    """
    for segment in audio_segments:
        if segment["start"] <= current_time <= segment["end"]:
            return segment["text"]
    return ""  # 如果找不到匹配的转录文本


def get_frames(video: cv2.VideoCapture):
    """ 
    获取视频的帧数据
    :return: (list[int], list[array]) (帧编号列表, 帧像素数据列表) 元组
    """
    frame_rate = round(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug(f"fps: {frame_rate} total: {total_frames}")
    ids, frames = [], []
    for current_frame in trange(
            0, total_frames, FRAME_INTERVAL * frame_rate, desc="当前进度", unit="frame"
    ):
        # 在 FRAME_INTERVAL 为 2（默认值），frame_rate 为 24
        # 即 FRAME_INTERVAL * frame_rate == 48 时测试
        # 直接设置当前帧的运行效率低于使用 grab 跳帧
        # 如果需要跳的帧足够多，也许直接设置效率更高
        # video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ids.append(current_frame // frame_rate)
        frames.append(frame)
        if len(frames) == SCAN_PROCESS_BATCH_SIZE:
            yield ids, frames
            ids = []
            frames = []
        for _ in range(FRAME_INTERVAL * frame_rate - 1):
            video.grab()  # 跳帧
    yield ids, frames


# process_assets.py

def process_video(path):
    """
    处理视频并返回处理完成的数据
    :param path: 视频路径
    :return: 生成器，每次生成 (frame_time, features, current_time, transcript) 的单个帧数据
    """
    logger.info(f"处理视频中：{path}")
    try:
        # 提取音频并转录
        audio_path = path.replace('.mp4', '.wav')
        extract_audio_from_video(path, audio_path)
        result = whisper_model.transcribe(audio_path)

        # 获取视频帧
        video = cv2.VideoCapture(path)
        frame_rate = round(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for current_frame in trange(0, total_frames, FRAME_INTERVAL * frame_rate, desc="Processing video frames"):
            # 定位到当前帧
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = video.read()
            if not ret:
                break

            # 提取帧特征
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feature = get_image_feature([frame])  # CLIP提取特征
            feature_bytes = feature.tobytes()     # 转为字节

            # 计算时间戳和转录文本
            current_time = current_frame / frame_rate
            transcript = get_audio_transcript_for_time(current_time, result["segments"])

            # 每次生成单个帧的数据
            yield (
                current_frame // frame_rate,  # frame_time (int)
                feature_bytes,                 # features (bytes)
                current_time,                  # 时间戳 (float)
                transcript                     # 文本 (str)
            )

        video.release()
    except Exception as e:
        logger.warning(f"处理视频出错：{path} {repr(e)}")
        traceback.print_exc()


def process_text(input_text):
    """
    预处理文字，返回文字特征
    :param input_text: string, 被处理的字符串
    :return: <class 'numpy.nparray'>, 文字特征
    """
    feature = None
    if not input_text:
        return None
    try:
        text = clip_processor(text=input_text, return_tensors="pt", padding=True).to(torch.device(DEVICE))
        feature = clip_model.get_text_features(text['input_ids']).detach().cpu().numpy()
    except Exception as e:
        logger.warning(f"处理文字报错：{repr(e)}")
        traceback.print_stack()
    return feature

def match_text_and_image_with_transcription(text_feature, image_feature, transcription):
    """
    匹配文字和图片，返回余弦相似度。这里的text_feature不仅包含转录文本，还可能包含查询文本。
    :param text_feature: <class 'numpy.nparray'>, 文字特征（包括转录文本）
    :param image_feature: <class 'numpy.nparray'>, 图片特征
    :param transcription: string, 转录文本
    :return: <class 'numpy.nparray'>, 文字和图片的余弦相似度，shape=(1, 1)
    """
    # 提取转录文本的特征
    transcription_feature = process_text(transcription)
    combined_text_feature = np.concatenate([text_feature, transcription_feature], axis=1)

    # 计算余弦相似度
    score = (image_feature @ combined_text_feature.T) / (
            np.linalg.norm(image_feature) * np.linalg.norm(combined_text_feature)
    )
    return score


def normalize_features(features):
    """
    归一化
    :param features: [<class 'numpy.nparray'>], 特征
    :return: <class 'numpy.nparray'>, 归一化后的特征
    """
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def multithread_normalize(features):
    """
    多线程执行归一化，只有对大矩阵效果才好
    :param features:  [<class 'numpy.nparray'>], 特征
    :return: <class 'numpy.nparray'>, 归一化后的特征
    """
    num_threads = os.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 将图像特征分成等分，每个线程处理一部分
        chunk_size = len(features) // num_threads
        chunks = [
            features[i: i + chunk_size] for i in range(0, len(features), chunk_size)
        ]
        # 并发执行特征归一化
        normalized_chunks = executor.map(normalize_features, chunks)
    # 将处理后的特征重新合并
    return np.concatenate(list(normalized_chunks))


def match_batch(
        positive_feature,
        negative_feature,
        image_features,
        positive_threshold,
        negative_threshold,
):
    """
    匹配image_feature列表并返回余弦相似度
    :param positive_feature: <class 'numpy.ndarray'>, 正向提示词特征
    :param negative_feature: <class 'numpy.ndarray'>, 反向提示词特征
    :param image_features: [<class 'numpy.ndarray'>], 图片特征列表
    :param positive_threshold: int/float, 正向提示分数阈值，高于此分数才显示
    :param negative_threshold: int/float, 反向提示分数阈值，低于此分数才显示
    :return: <class 'numpy.nparray'>, 提示词和每个图片余弦相似度列表，shape=(n, )，如果小于正向提示分数阈值或大于反向提示分数阈值则会置0
    """
    # 计算余弦相似度
    if len(image_features) > 1024:  # 多线程只对大矩阵效果好，1024是随便写的
        new_features = multithread_normalize(image_features)
    else:
        new_features = normalize_features(image_features)
    if positive_feature is None: # 没有正向feature就把分数全部设成1
        positive_scores = np.ones(len(new_features))
    else:
        new_text_positive_feature = positive_feature / np.linalg.norm(positive_feature)
        positive_scores = (new_features @ new_text_positive_feature.T).squeeze(-1)
    if negative_feature is not None:
        new_text_negative_feature = negative_feature / np.linalg.norm(negative_feature)
        negative_scores = (new_features @ new_text_negative_feature.T).squeeze(-1)
    # 根据阈值进行过滤
    scores = np.where(positive_scores < positive_threshold / 100, 0, positive_scores)
    if negative_feature is not None:
        scores = np.where(negative_scores > negative_threshold / 100, 0, scores)
    return scores
