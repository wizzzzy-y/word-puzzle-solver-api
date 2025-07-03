import cv2
import numpy as np
import pytesseract
from PIL import Image
import itertools
import os
import logging

logger = logging.getLogger(__name__)

class WordPuzzleSolver:
    def __init__(self):
        self.dictionary = self._load_dictionary()
        self.tesseract_config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        logger.info(f"Solver initialized with {len(self.dictionary)} words")
    
    def _load_dictionary(self):
        words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD',
            'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS',
            'CAR', 'CAT', 'RAT', 'BAT', 'HAT', 'MAT', 'PAT', 'SAT', 'FAT', 'VAT',
            'PARK', 'CARD', 'PACK', 'RACK', 'CRAP', 'CARP', 'DARK', 'MARK', 'BARK ',
            'ARC', 'PAR', 'RAP', 'CAP', 'TAP', 'RAT', 'PAT', 'BAT', 'CAT', 'HAT',
            'APE', 'ARE', 'EAR', 'ERA', 'PEA', 'REP', 'JAR', 'JET', 'JOB', 'JOY',
            'LAD', 'LAP', 'LAY', 'LET', 'LID', 'LIP', 'LOT', 'LOW', 'LAW'
        }
        return words
    
    def solve_puzzle(self, image_path):
        try:
            letters_data = self._detect_letters(image_path)
            if not letters_data:
                return {"swipes": []}
            
            swipes = self._generate_word_swipes(letters_data)
            return {"swipes": swipes}
            
        except Exception as e:
            logger.error(f"Solver error: {e}")
            return {"swipes": []}
    
    def _detect_letters(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            h, w = img.shape[:2]
            
            # Focus on bottom 25% for letter wheel
            wheel_start_y = int(h * 0.75)
            center_x = w // 2
            wheel_radius = min(w, h) // 4
            wheel_start_x = max(0, center_x - wheel_radius)
            wheel_end_x = min(w, center_x + wheel_radius)
            
            wheel = img[wheel_start_y: h, wheel_start_x:wheel_end_x]
            
            # Preprocess
            gray = cv2.cvtColor(wheel, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            letters = []
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(contour)
                if area < 200 or area > wheel.shape[0] * wheel.shape[1] * 0.3:
                    continue
                
                x, y, w_box, h_box = cv2.boundingRect(contour)
                if w_box < 20 or h_box < 20:
                    continue
                
                letter_region = wheel[y:y+h_box, x:x+w_box]
                letter = self._ocr_letter(letter_region)
                
                if letter and letter.isalpha():
                    center_x_abs = wheel_start_x + x + w_box // 2
                    center_y_abs = wheel_start_y + y + h_box // 2
                    
                    letters.append({
                        'letter': letter.upper(),
                        'position': [center_x_abs, center_y_abs]
                    })
                
                if len(letters) >= 6:
                    break
            
            return letters
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _ocr_letter(self, image_region):
        try:
            if image_region.size == 0:
                return ""
            
            # Convert to PIL
            if len(image_region.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image_region)
            
            # Resize if too small
            if pil_img.size[0] < 20 or pil_img.size[1] < 20:
                pil_img = pil_img.resize((40, 40))
            
            # OCR
            text = pytesseract.image_to_string(pil_img, config=self.tesseract_config).strip()
            
            for char in text:
                if char.isalpha():
                    return char.upper()
            
            return ""
        except:
            return ""
    
    def _generate_word_swipes(self, letters_data):
        try:
            if not letters_data:
                return []
            
            letter_positions = {}
            for item in letters_data:
                letter = item['letter']
                pos = item['position']
                
                if letter not in letter_positions:
                    letter_positions[letter] = []
                letter_positions[letter].append(pos)
            
            available_letters = list(letter_positions.keys())
            swipes = []
            
            # Generate words of different lengths
            for length in range(3, min(6, len(available_letters) + 1)):
                for perm in itertools.permutations(available_letters, length):
                    word = ''.join(perm)
                    
                    if word in self.dictionary:
                        path = []
                        used_positions = set()
                        
                        valid_path = True
                        for letter in perm:
                            available_pos = [pos for pos in letter_positions[letter] 
                                             if tuple(pos) not in used_positions]
                            
                            if not available_pos:
                                valid_path = False
                                break
                            
                            position = available_pos[0]
                            path.append(position)
                            used_positions.add(tuple(position))
                        
                        if valid_path and path:
                            swipes.append(path)
                    
                    if len(swipes) >= 10:
                        break
                
                if len(swipes) >= 10:
                    break
            
            return swipes[:10]
            
        except Exception as e:
            logger.error(f"Word generation error: {e}")
            return []

def solve_word_puzzle(image_path):
    solver = WordPuzzleSolver()
    return solver.solve_puzzle(image_path)