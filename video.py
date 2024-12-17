import os
import json
import time
import sys
import urllib.request
from multiprocessing.dummy import Pool
import cv2
import shutil

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

import random

import logging
logging.basicConfig(filename='download_{}.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# Set this to youtube-dl if you want to use youtube-dl.
# The the README for an explanation regarding yt-dlp vs youtube-dl.
youtube_downloader = "yt-dlp"

download_words = ["a", "b", "d", "e", "f"]

def save_video(data, saveto):
    with open(saveto, 'wb+') as f:
        f.write(data)

def request_video(url, referer=''):
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

    headers = {'User-Agent': user_agent,
               }
    
    if referer:
        headers['Referer'] = referer

    request = urllib.request.Request(url, None, headers)  # The assembled request

    logging.info('Requesting {}'.format(url))
    response = urllib.request.urlopen(request)
    data = response.read()  # The data you need

    return data

def download_others(url, dirname, video_id):
    saveto = os.path.join(dirname, '{}.mp4'.format(video_id))
    if os.path.exists(saveto):
        logging.info('{} exists at {}'.format(video_id, saveto))
        return 
    
    data = request_video(url)
    save_video(data, saveto)

def download_yt_videos(url, dirname, video_id): # indexfile, saveto='raw_videos'
    if os.path.exists(os.path.join(dirname, video_id + '.mp4')) or os.path.exists(os.path.join(dirname, video_id + '.mkv')):
        logging.info('YouTube videos {} already exists.'.format(url))
    else:
        cmd = f"{youtube_downloader} \"{{}}\" -o \"{{}}{video_id}.%(ext)s\""
        cmd = cmd.format(url, dirname + os.path.sep)

        rv = os.system(cmd)
        
        if not rv:
            logging.info('Finish downloading youtube video url {}'.format(url))
        else:
            logging.error('Unsuccessful downloading - youtube video url {}'.format(url))

        # please be nice to the host - take pauses and avoid spamming
        time.sleep(random.uniform(1.0, 1.5))

            
def download_aslpro(url, dirname, video_id):
    saveto = os.path.join(dirname, '{}.swf'.format(video_id))
    if os.path.exists(saveto):
        logging.info('{} exists at {}'.format(video_id, saveto))
        return 

    data = request_video(url, referer='http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi')
    save_video(data, saveto)

def select_download_method(url):
    if 'aslpro' in url:
        return download_aslpro
    elif 'youtube' in url or 'youtu.be' in url:
        return download_yt_videos
    else:
        return download_others

def trim_video(input_path, output_path, start_frame, end_frame, codec):
    # Load the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open the input video file.")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*codec)

    if start_frame >= total_frames or end_frame > total_frames:
        print("Error: Start or end time exceeds video duration.")
        return False

    if start_frame >= end_frame:
        print("Error: Start time must be less than end time.")
        return False

    print(f"Trimming video from frame {start_frame} to {end_frame} (FPS: {fps}).")

    # Create VideoWriter
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print("Error: Cannot open the output video file for writing.")
        return False

    # Read and write frames
    frame_count = 0
    frames_written = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        if start_frame <= frame_count < end_frame:
            out.write(frame)
            frames_written += 1

        frame_count += 1
        if frame_count >= end_frame:
            break

    print(f"Frames written: {frames_written}")
    cap.release()
    out.release()

    if frames_written == 0:
        print("Error: No frames were written to the output file.")
        return False

    print(f"Video trimmed successfully! Saved to {output_path}")

def process_json_array(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    for item in data:
        if item["gloss"] in download_words:
            gloss = item["gloss"]
            instances = item["instances"]
            saveto = "videos"

            if not os.path.exists(gloss):
                os.makedirs(gloss)

            for inst in instances:
                video_url = inst['url']
                video_id = inst['video_id']
                start_frame = inst['frame_start']
                end_frame = inst['frame_end']
                frame_rate = inst['fps']

                logging.info('gloss: {}, video: {}.'.format(gloss, video_id))

                download_method = select_download_method(video_url)    

                try:
                    download_method(video_url, saveto, video_id) # Video formats: mp4, mkv, swf

                    input_path = ""
                    output_path = ""
                    codec = ""
                    if os.path.exists(os.path.join(saveto, '{}.swf'.format(video_id))):
                        input_path = os.path.join(saveto, '{}.swf'.format(video_id))
                        output_path = os.path.join(gloss, '{}.swf'.format(video_id))
                        codec = 'FLV1'
                    elif os.path.exists(os.path.join(saveto, '{}.mp4'.format(video_id))):
                        input_path = os.path.join(saveto, '{}.mp4'.format(video_id))
                        output_path = os.path.join(gloss, '{}.mp4'.format(video_id))
                        codec = "mp4v"
                    elif os.path.exists(os.path.join(saveto, '{}.mkv'.format(video_id))):
                        input_path = os.path.join(saveto, '{}.mkv'.format(video_id))
                        output_path = os.path.join(gloss, '{}.mkv'.format(video_id))
                        codec = 'X264'

                    if start_frame != 1 | end_frame != -1:                        
                        print(F"[{video_id}]: Splitting video at {input_path}")
                        trim_video(input_path, output_path, start_frame, end_frame, codec)
                    else:
                        shutil.copy2(input_path, output_path)
    
                except Exception as e:
                    logging.error('Unsuccessful downloading - video {}'.format(video_id))

def check_youtube_dl_version():
    ver = os.popen(f'{youtube_downloader} --version').read()

    assert ver, f"{youtube_downloader} cannot be found in PATH. Please verify your installation."


check_youtube_dl_version()
process_json_array('WLASL_v0.3.json')
