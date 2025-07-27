
import os
import re
from datetime import timedelta
import yt_dlp
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx.all import resize, crop
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    url: str
    language: str = "en"
    segment_type: str = "2"

class YouTubeShortGenerator:
    def __init__(self):
        self.video_url = ""
        self.video_id = ""
        self.video_title = "YouTube Video"
        self.video_length = 0
        self.output_folder = "generated_shorts"
        self.max_short_length = 60
        self.min_short_length = 30
        self.engagement_data = []
        self.short_ratio = (9, 16)
        self.transcript = []
        self.language = 'en'
        self.font_mapping = {'en': 'Arial', 'hi': 'DejaVu-Sans'}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def get_video_info(self) -> bool:
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'extract_flat': True,
                'username': os.getenv('YT_USERNAME', ''),
                'password': os.getenv('YT_PASSWORD', ''),
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'cookies': os.getenv('YT_COOKIES', '')
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.video_url, download=False)
                self.video_title = info.get('title', 'YouTube Video')
                self.video_length = info.get('duration', 0)
                self.video_id = info.get('id', '')
                if self.video_length < self.min_short_length:
                    logger.error(f"Video too short: {self.video_length}s < {self.min_short_length}s")
                    return False
                return True
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return False

    def download_video(self) -> str:
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            temp_path = os.path.join(self.output_folder, f"temp_{self.video_id}.mp4")
            ydl_opts = {
                'format': 'best[height<=480]',
                'outtmpl': temp_path,
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.video_url])
            return temp_path
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return ""

    def get_transcript(self) -> bool:
        try:
            languages_to_try = ['en', 'hi'] if self.language == 'hi' else ['hi', 'en']
            for lang in languages_to_try:
                try:
                    self.transcript = YouTubeTranscriptApi.get_transcript(self.video_id, languages=[lang])
                    self.language = lang
                    logger.info(f"Found transcript in {lang}")
                    return True
                except NoTranscriptFound:
                    continue
            try:
                self.transcript = YouTubeTranscriptApi.get_transcript(self.video_id)
                return True
            except:
                logger.warning("No transcript available")
                return False
        except TranscriptsDisabled:
            logger.warning("Transcripts disabled")
            return False
        except Exception as e:
            logger.error(f"Error fetching transcript: {e}")
            return False

    def analyze_engagement(self, video_path):
        try:
            if not self.transcript:
                return self.create_even_segments()
            segments = []
            current_segment = None
            for entry in self.transcript:
                start = entry['start']
                end = start + entry['duration']
                text = entry['text']
                word_count = len(text.split())
                density = word_count / entry['duration'] if entry['duration'] > 0 else 0
                if current_segment is None:
                    current_segment = {'start': start, 'end': end, 'text': text, 'score': density * 10}
                else:
                    potential_end = end
                    if potential_end - current_segment['start'] <= self.max_short_length:
                        current_segment['end'] = potential_end
                        current_segment['text'] += " " + text
                        current_segment['score'] += density * 10
                    else:
                        if current_segment['end'] - current_segment['start'] >= self.min_short_length:
                            segments.append(current_segment)
                        current_segment = {'start': start, 'end': end, 'text': text, 'score': density * 10}
            if current_segment and current_segment['end'] - current_segment['start'] >= self.min_short_length:
                segments.append(current_segment)
            segments.sort(key=lambda x: x['score'], reverse=True)
            self.engagement_data = segments[:3]
            if not self.engagement_data:
                return self.create_even_segments()
            return True
        except Exception as e:
            logger.error(f"Error analyzing engagement: {e}")
            return False

    def create_even_segments(self):
        try:
            self.engagement_data = []
            segment_count = min(3, int(self.video_length // self.min_short_length))
            for i in range(segment_count):
                start = i * self.min_short_length
                end = start + self.min_short_length
                if end > self.video_length:
                    end = self.video_length
                    start = max(0, end - self.min_short_length)
                self.engagement_data.append({'start': start, 'end': end, 'score': 80})
            return True
        except Exception as e:
            logger.error(f"Error creating even segments: {e}")
            return False

    def convert_to_shorts_format(self, clip):
        try:
            width, height = clip.size
            target_ratio = self.short_ratio[0] / self.short_ratio[1]
            if width/height > target_ratio:
                new_width = height * target_ratio
                x_center = width / 2
                return crop(clip, x1=x_center - new_width/2, x2=x_center + new_width/2)
            else:
                new_height = width / target_ratio
                y_center = height / 2
                return crop(clip, y1=y_center - new_height/2, y2=y_center + new_height/2)
        except Exception as e:
            logger.error(f"Error converting to shorts format: {e}")
            return clip

    def detect_faces_in_frame(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
            return faces
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def get_safe_text_position(self, clip, start_time, duration):
        try:
            sample_times = np.linspace(start_time, start_time + duration, num=3)
            face_positions = []
            for t in sample_times:
                frame = clip.get_frame(t)
                faces = self.detect_faces_in_frame(frame)
                for (x, y, w, h) in faces:
                    rel_x = (x + w/2) / clip.size[0]
                    rel_y = (y + h/2) / clip.size[1]
                    face_positions.append((rel_x, rel_y))
            if not face_positions:
                return 0.85
            avg_y = sum(pos[1] for pos in face_positions) / len(face_positions)
            return max(0.75, min(0.9, avg_y - 0.2))
        except Exception as e:
            logger.error(f"Error determining text position: {e}")
            return 0.85

    def add_transcript_overlay(self, clip, start_time):
        try:
            if not self.transcript:
                return clip
            clip_end = start_time + clip.duration
            relevant_segments = [seg for seg in self.transcript if seg['start'] < clip_end and (seg['start'] + seg['duration']) > start_time]
            font = self.font_mapping.get(self.language, 'Arial')
            text_y_position = self.get_safe_text_position(clip, start_time, clip.duration)
            subtitles = []
            for seg in relevant_segments:
                seg_start = max(0, seg['start'] - start_time)
                seg_end = min(clip.duration, (seg['start'] + seg['duration']) - start_time)
                try:
                    txt_clip = (TextClip(seg['text'], fontsize=30, color='white', bg_color='rgba(0,0,0,0.7)', size=(clip.size[0]*0.85, None), method='caption', font=font, align='center', stroke_color='black', stroke_width=1)
                    .set_position(('center', text_y_position), relative=True)
                    .set_start(seg_start)
                    .set_duration(seg_end - seg_start))
                    subtitles.append(txt_clip)
                except Exception as e:
                    logger.error(f"Error creating text clip: {e}")
                    continue
            if subtitles:
                return CompositeVideoClip([clip] + subtitles)
            return clip
        except Exception as e:
            logger.error(f"Error adding transcript overlay: {e}")
            return clip

    def generate_short(self, start_time, end_time, clip_num):
        try:
            video_path = os.path.join(self.output_folder, f"temp_{self.video_id}.mp4")
            if not os.path.exists(video_path):
                video_path = self.download_video()
                if not video_path:
                    return ""
            actual_start = max(0, start_time)
            actual_end = min(end_time, self.video_length)
            if actual_end - actual_start < self.min_short_length:
                actual_end = min(actual_start + self.min_short_length, self.video_length)
                if actual_end - actual_start < self.min_short_length:
                    actual_start = max(0, actual_end - self.min_short_length)
            logger.info(f"Creating short {clip_num} from {actual_start:.1f}s to {actual_end:.1f}s")
            with VideoFileClip(video_path) as clip:
                subclip = clip.subclip(actual_start, actual_end)
                vertical_clip = self.convert_to_shorts_format(subclip)
                final_clip = self.add_transcript_overlay(vertical_clip, actual_start)
                output_path = os.path.join(self.output_folder, f"short_{self.video_id}_{clip_num}.mp4")
                final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', threads=2, fps=24, preset='ultrafast', logger=None)
                return output_path
        except Exception as e:
            logger.error(f"Error generating short: {e}")
            return ""

    async def process_video(self, url, language, segment_type):
        self.video_url = url
        self.language = language.lower()
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        self.video_id = match.group(1)
        if not self.get_video_info():
            raise HTTPException(status_code=400, detail="Failed to get video info")
        self.get_transcript()
        video_path = self.download_video()
        if not video_path:
            raise HTTPException(status_code=500, detail="Failed to download video")
        if segment_type == '1':
            if not self.analyze_engagement(video_path):
                raise HTTPException(status_code=500, detail="Failed to analyze engagement")
        else:
            if not self.create_even_segments():
                raise HTTPException(status_code=500, detail="Failed to create segments")
        results = []
        for i, segment in enumerate(self.engagement_data[:3], 1):
            output_path = self.generate_short(segment['start'], segment['end'], i)
            if output_path:
                results.append(output_path)
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as e:
            logger.warning(f"Could not remove temp file {video_path}: {e}")
        return {"status": "success", "shorts": results}

@app.post("/generate_shorts")
async def generate_shorts(request: VideoRequest):
    generator = YouTubeShortGenerator()
    try:
        result = await generator.process_video(request.url, request.language, request.segment_type)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
