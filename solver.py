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
            # Words from P, R, T, A
            'PAR', 'RAP', 'TAP', 'RAT', 'PAT', 'ART', 'TAR', 'APT',
            'PART', 'TRAP', 'TARP', 'RAPT', 'PRAT',
            
            # Common 3-letter words
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD',
            'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS',
            'CAR', 'CAT', 'RAT', 'BAT', 'HAT', 'MAT', 'PAT', 'SAT', 'FAT', 'VAT',
            'ARC', 'CAP', 'TAP', 'APE', 'EAR', 'ERA', 'PEA', 'REP', 'JAR', 'JET', 
            'JOB', 'JOY', 'LAD', 'LAP', 'LAY', 'LET', 'LID', 'LIP', 'LOT', 'LOW', 'LAW',
            
            # Common 4+ letter words  
            'PARK', 'CARD', 'PACK', 'RACK', 'CRAP', 'CARP', 'DARK', 'MARK', 'BARK'
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
            
            # Look for the circular letter wheel - it's typically in the bottom half
            # Start from bottom 40% of the image
            search_start_y = int(h * 0.6)
            search_region = img[search_start_y:h, :]
            
            # Convert to grayscale for circle detection
            gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
            
            # Detect circles using HoughCircles
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                                     param1=50, param2=30, minRadius=80, maxRadius=200)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # Use the largest circle (likely the letter wheel)
                if len(circles) > 0:
                    # Sort by radius and take the largest
                    circle = max(circles, key=lambda c: c[2])
                    cx, cy, radius = circle
                    
                    # Adjust coordinates to full image
                    center_x = cx
                    center_y = cy + search_start_y
                    
                    # Extract the circular region with some padding
                    padding = 20
                    x1 = max(0, center_x - radius - padding)
                    y1 = max(0, center_y - radius - padding)
                    x2 = min(w, center_x + radius + padding)
                    y2 = min(h, center_y + radius + padding)
                    
                    wheel_region = img[y1:y2, x1:x2]
                    
                    # Detect letters in this circular region
                    letters = self._detect_letters_in_circle(wheel_region, 
                                                           center_x - x1, 
                                                           center_y - y1, 
                                                           radius, x1, y1)
                    if letters:
                        return letters
            
            # Fallback: Look for letters in the bottom third using contour detection
            return self._detect_letters_fallback(img)
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _detect_letters_in_circle(self, wheel_img, center_x, center_y, radius, offset_x, offset_y):
        """Detect letters arranged in a circle"""
        letters = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(wheel_img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter contours by size - letters should be reasonably sized
            if area < 100 or area > 5000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if contour is roughly letter-sized
            if w < 15 or h < 15 or w > 100 or h > 100:
                continue
            
            # Check aspect ratio (letters shouldn't be too wide or too tall)
            aspect_ratio = w / h
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            # Extract letter region
            letter_region = wheel_img[y:y+h, x:x+w]
            
            # OCR the letter
            letter = self._ocr_letter(letter_region)
            
            if letter and letter.isalpha():
                # Calculate absolute position
                letter_center_x = offset_x + x + w // 2
                letter_center_y = offset_y + y + h // 2
                
                letters.append({
                    'letter': letter.upper(),
                    'position': [int(letter_center_x), int(letter_center_y)]
                })
        
        return letters
    
    def _detect_letters_fallback(self, img):
        """Fallback method for letter detection"""
        h, w = img.shape[:2]
        
        # Focus on bottom 30% of image
        wheel_start_y = int(h * 0.7)
        wheel_region = img[wheel_start_y:h, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(wheel_region, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPT_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        letters = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < 100 or area > 3000:
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(contour)
            if w_box < 15 or h_box < 15:
                continue
            
            letter_region = wheel_region[y:y+h_box, x:x+w_box]
            letter = self._ocr_letter(letter_region)
            
            if letter and letter.isalpha():
                center_x_abs = x + w_box // 2
                center_y_abs = wheel_start_y + y + h_box // 2
                
                letters.append({
                    'letter': letter.upper(),
                    'position': [center_x_abs, center_y_abs]
                })
            
            if len(letters) >= 6:
                break
        
        return letters
    
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