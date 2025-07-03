import cv2
import numpy as np
import pytesseract
from PIL import Image
import itertools
import os
import logging
import requests
import time

logger = logging.getLogger(__name__)

class WordPuzzleSolver:
    def __init__(self):
        self.dictionary = self._load_dictionary()
        self.tesseract_config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        logger.info(f"Solver initialized with {len(self.dictionary)} words")
    
    def _load_dictionary(self):
        """Load comprehensive English dictionary from online source"""
        # Try to load from cache first
        cache_file = "english_words_cache.txt"
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    words = set(word.strip().upper() for word in f.readlines() if word.strip())
                logger.info(f"Loaded {len(words)} words from cache")
                return words
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Download comprehensive word list from GitHub
        try:
            logger.info("Downloading comprehensive English dictionary...")
            url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
            
            # Set timeout for production environment
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Process word list
            words_text = response.text
            words = set()
            
            for word in words_text.strip().split('\n'):
                word = word.strip().upper()
                if word and len(word) >= 2 and word.isalpha():  # Filter valid words
                    words.add(word)
            
            # Cache the downloaded words
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    for word in sorted(words):
                        f.write(word + '\n')
                logger.info(f"Cached {len(words)} words to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache words: {e}")
            
            logger.info(f"Downloaded {len(words)} words from online source")
            return words
            
        except Exception as e:
            logger.error(f"Failed to download dictionary: {e}")
            
            # Fallback to basic word set
            logger.info("Using fallback dictionary")
            return self._get_fallback_dictionary()
    
    def _get_fallback_dictionary(self):
        """Fallback dictionary when online download fails"""
        return {
            # Most common English words - comprehensive fallback
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'WAY', 'WHO', 'OIL', 'SIT', 'SET',
            
            # 3-letter words
            'ACE', 'ACT', 'ADD', 'AGE', 'AID', 'AIM', 'AIR', 'ARM', 'ART', 'ASK', 'ATE', 'BAD', 'BAG', 'BAR', 'BAT', 'BED', 'BEE', 'BET', 'BIG', 'BIT', 'BOX', 'BUY', 'CAR', 'CAT', 'COW', 'CRY', 'CUP', 'CUT', 'DOG', 'EAR', 'EAT', 'EGG', 'END', 'EYE', 'FAR', 'FEW', 'FLY', 'FOR', 'FUN', 'GOT', 'GUN', 'GUY', 'HAT', 'HIT', 'HOT', 'JOB', 'LAW', 'LAY', 'LEG', 'LET', 'LIE', 'LOT', 'LOW', 'MAN', 'MAP', 'NET', 'PAY', 'PEN', 'PET', 'PIG', 'POT', 'RAT', 'RED', 'RUN', 'SAD', 'SAT', 'SEA', 'SIT', 'SIX', 'SKY', 'SUN', 'TAX', 'TEA', 'TEN', 'TOP', 'TRY', 'VAN', 'WAR', 'WET', 'WIN', 'YES', 'YET', 'ZOO',
            'FAT', 'FIT', 'FIX', 'FOG', 'FOX', 'FUR', 'GAP', 'GAS', 'GEL', 'GEM', 'GOD', 'ICE', 'ILL', 'INK', 'JAM', 'JAW', 'JOG', 'JOT', 'JOY', 'KEY', 'KID', 'KIT', 'LAB', 'LAG', 'LIT', 'LOG', 'MAD', 'MIX', 'MOB', 'MUD', 'NUT', 'ODD', 'ORB', 'OWL', 'OWN', 'PAD', 'PAN', 'PAR', 'PAW', 'PIN', 'PIT', 'PLY', 'POD', 'PRO', 'PUB', 'PUN', 'PUP', 'RAG', 'RAM', 'RAP', 'RAW', 'RAY', 'RIB', 'RID', 'RIM', 'RIP', 'ROB', 'ROD', 'ROT', 'ROW', 'RUB', 'RUG', 'RUM', 'RUT', 'SAP', 'SAW', 'SIN', 'SIP', 'SIR', 'SOB', 'SOD', 'SON', 'SOP', 'SOW', 'SOY', 'SPA', 'SPY', 'STY', 'TAB', 'TAG', 'TAN', 'TAP', 'TAR', 'TAT', 'TIC', 'TIE', 'TIN', 'TIP', 'TOE', 'TON', 'TOT', 'TOW', 'TOY', 'TUB', 'TUG', 'TUT', 'URN', 'VIA', 'VIE', 'VOW', 'WAD', 'WAG', 'WAN', 'WAX', 'WEB', 'WED', 'WEE', 'WIG', 'WIT', 'WOE', 'WOK', 'WON', 'WOO', 'WOW', 'YAK', 'YAM', 'YAP', 'YAW', 'YEA', 'YEN', 'YEP', 'YEW', 'YIN', 'YIP', 'YON', 'ZAP', 'ZED', 'ZEE', 'ZEN', 'ZIP', 'ZIT',
            
            # 4+ letter words (common ones)
            'ABLE', 'AREA', 'ARMY', 'BABY', 'BACK', 'BALL', 'BAND', 'BANK', 'BASE', 'BATH', 'BEAR', 'BEAT', 'BEEN', 'BELL', 'BEST', 'BILL', 'BIRD', 'BLOW', 'BLUE', 'BOAT', 'BODY', 'BONE', 'BOOK', 'BORN', 'BOTH', 'BOYS', 'BUSY', 'CALL', 'CAME', 'CAMP', 'CARD', 'CARE', 'CARS', 'CASE', 'CASH', 'CELL', 'CITY', 'CLUB', 'COAL', 'COAT', 'COLD', 'COME', 'COOK', 'COOL', 'COPY', 'CORN', 'COST', 'CREW', 'DARK', 'DATA', 'DATE', 'DAYS', 'DEAD', 'DEAL', 'DEAR', 'DEEP', 'DESK', 'DOES', 'DONE', 'DOOR', 'DOWN', 'DRAW', 'DREW', 'DROP', 'DRUG', 'EACH', 'EARN', 'EAST', 'EASY', 'EDGE', 'ELSE', 'EVEN', 'EVER', 'FACE', 'FACT', 'FAIL', 'FAIR', 'FALL', 'FARM', 'FAST', 'FEAR', 'FEEL', 'FEET', 'FELL', 'FELT', 'FILE', 'FILL', 'FILM', 'FIND', 'FINE', 'FIRE', 'FIRM', 'FISH', 'FIVE', 'FLAT', 'FLOW', 'FOOD', 'FOOT', 'FORM', 'FOUR', 'FREE', 'FROM', 'FULL', 'FUND', 'GAME', 'GAVE', 'GIRL', 'GIVE', 'GLAD', 'GOES', 'GOLD', 'GONE', 'GOOD', 'GREW', 'GROW', 'HAIR', 'HALF', 'HALL', 'HAND', 'HARD', 'HARM', 'HEAD', 'HEAR', 'HEAT', 'HELD', 'HELP', 'HERE', 'HIGH', 'HILL', 'HOLD', 'HOME', 'HOPE', 'HOUR', 'HUGE', 'IDEA', 'INTO', 'ITEM', 'JOBS', 'JOIN', 'JUMP', 'JUST', 'KEEP', 'KEPT', 'KIND', 'KING', 'KNEW', 'KNOW', 'LAND', 'LAST', 'LATE', 'LEAD', 'LEFT', 'LESS', 'LIFE', 'LIKE', 'LINE', 'LIST', 'LIVE', 'LOAN', 'LONG', 'LOOK', 'LORD', 'LOSE', 'LOSS', 'LOST', 'LOVE', 'MADE', 'MAIL', 'MAIN', 'MAKE', 'MALE', 'MANY', 'MARK', 'MASS', 'MEAT', 'MEET', 'MIND', 'MINE', 'MISS', 'MODE', 'MORE', 'MOST', 'MOVE', 'MUCH', 'MUST', 'NAME', 'NEAR', 'NECK', 'NEED', 'NEWS', 'NEXT', 'NICE', 'NINE', 'NODE', 'NONE', 'NOON', 'NOTE', 'OPEN', 'ORAL', 'OVER', 'PAGE', 'PAID', 'PAIN', 'PAIR', 'PARK', 'PART', 'PASS', 'PAST', 'PATH', 'PEAK', 'PICK', 'PINK', 'PLAN', 'PLAY', 'PLOT', 'PLUS', 'POLL', 'POOL', 'POOR', 'PORT', 'POST', 'PULL', 'PURE', 'PUSH', 'RACE', 'RAIN', 'RANK', 'RATE', 'READ', 'REAL', 'REAR', 'RELY', 'REST', 'RICH', 'RIDE', 'RING', 'RISE', 'RISK', 'ROAD', 'ROCK', 'ROLE', 'ROLL', 'ROOM', 'ROOT', 'ROSE', 'RULE', 'RUNS', 'SAFE', 'SAID', 'SALE', 'SAME', 'SAVE', 'SEAT', 'SEEM', 'SELF', 'SELL', 'SEND', 'SENT', 'SHIP', 'SHOP', 'SHOT', 'SHOW', 'SICK', 'SIDE', 'SIGN', 'SITE', 'SIZE', 'SKIN', 'SLIP', 'SLOW', 'SNOW', 'SOFT', 'SOIL', 'SOLD', 'SOME', 'SONG', 'SOON', 'SORT', 'SOUL', 'SPOT', 'STAR', 'STAY', 'STEP', 'STOP', 'SUCH', 'SURE', 'TAKE', 'TALK', 'TALL', 'TANK', 'TAPE', 'TASK', 'TEAM', 'TELL', 'TERM', 'TEST', 'TEXT', 'THAN', 'THAT', 'THEN', 'THEY', 'THIN', 'THIS', 'TIME', 'TOLD', 'TONE', 'TOOK', 'TOOL', 'TOUR', 'TOWN', 'TREE', 'TRUE', 'TURN', 'TYPE', 'UNIT', 'UPON', 'USED', 'USER', 'VARY', 'VAST', 'VERY', 'VIEW', 'VOTE', 'WAGE', 'WAIT', 'WAKE', 'WALK', 'WALL', 'WANT', 'WARD', 'WARM', 'WASH', 'WAVE', 'WAYS', 'WEAK', 'WEAR', 'WEEK', 'WELL', 'WENT', 'WERE', 'WEST', 'WHAT', 'WHEN', 'WIDE', 'WIFE', 'WILD', 'WILL', 'WIND', 'WINE', 'WING', 'WIRE', 'WISE', 'WISH', 'WITH', 'WOOD', 'WORD', 'WORE', 'WORK', 'YARD', 'YEAH', 'YEAR', 'YOUR', 'ZERO', 'ZONE',
            
            # Common 5+ letter words
            'ABOUT', 'ABOVE', 'ABUSE', 'ACTOR', 'ACUTE', 'ADMIT', 'ADOPT', 'ADULT', 'AFTER', 'AGAIN', 'AGENT', 'AGREE', 'AHEAD', 'ALARM', 'ALBUM', 'ALERT', 'ALIEN', 'ALIGN', 'ALIKE', 'ALIVE', 'ALLOW', 'ALONE', 'ALONG', 'ALTER', 'ANGLE', 'ANGRY', 'APART', 'APPLE', 'APPLY', 'ARENA', 'ARGUE', 'ARISE', 'ARRAY', 'ASIDE', 'ASSET', 'AVOID', 'AWAKE', 'AWARD', 'AWARE', 'BADLY', 'BASIC', 'BEACH', 'BEGAN', 'BEGIN', 'BEING', 'BENCH', 'BIRTH', 'BLACK', 'BLAME', 'BLANK', 'BLIND', 'BLOCK', 'BLOOD', 'BOARD', 'BOOST', 'BOOTH', 'BOUND', 'BRAIN', 'BRAND', 'BREAD', 'BREAK', 'BREED', 'BRIEF', 'BRING', 'BROAD', 'BROKE', 'BROWN', 'BUILD', 'BUILT', 'CATCH', 'CAUSE', 'CHAIN', 'CHAIR', 'CHAOS', 'CHARM', 'CHART', 'CHASE', 'CHEAP', 'CHECK', 'CHEST', 'CHIEF', 'CHILD', 'CHINA', 'CHOSE', 'CIVIL', 'CLAIM', 'CLASS', 'CLEAN', 'CLEAR', 'CLICK', 'CLIMB', 'CLOCK', 'CLOSE', 'CLOUD', 'COACH', 'COAST', 'COULD', 'COUNT', 'COURT', 'COVER', 'CRAFT', 'CRASH', 'CRAZY', 'CREAM', 'CRIME', 'CROSS', 'CROWD', 'CROWN', 'CRUDE', 'CURVE', 'CYCLE', 'DAILY', 'DANCE', 'DATED', 'DEALT', 'DEATH', 'DEBUT', 'DELAY', 'DEPTH', 'DOING', 'DOUBT', 'DOZEN', 'DRAFT', 'DRAMA', 'DRANK', 'DREAM', 'DRESS', 'DRILL', 'DRINK', 'DRIVE', 'DROVE', 'DYING', 'EAGER', 'EARLY', 'EARTH', 'EIGHT', 'ELITE', 'EMPTY', 'ENEMY', 'ENJOY', 'ENTER', 'ENTRY', 'EQUAL', 'ERROR', 'EVENT', 'EVERY', 'EXACT', 'EXIST', 'EXTRA', 'FAITH', 'FALSE', 'FAULT', 'FIELD', 'FIFTH', 'FIFTY', 'FIGHT', 'FINAL', 'FIRST', 'FIXED', 'FLASH', 'FLEET', 'FLOOR', 'FLUID', 'FOCUS', 'FORCE', 'FORTH', 'FORTY', 'FORUM', 'FOUND', 'FRAME', 'FRANK', 'FRAUD', 'FRESH', 'FRONT', 'FRUIT', 'FULLY', 'FUNNY', 'GIANT', 'GIVEN', 'GLASS', 'GLOBE', 'GOING', 'GRACE', 'GRADE', 'GRAND', 'GRANT', 'GRASS', 'GRAVE', 'GREAT', 'GREEN', 'GROSS', 'GROUP', 'GROWN', 'GUARD', 'GUESS', 'GUEST', 'GUIDE', 'HAPPY', 'HARSH', 'HEART', 'HEAVY', 'HENCE', 'HORSE', 'HOTEL', 'HOUSE', 'HUMAN', 'IDEAL', 'IMAGE', 'INDEX', 'INNER', 'INPUT', 'ISSUE', 'JAPAN', 'JOINT', 'JUDGE', 'KNOWN', 'LABEL', 'LARGE', 'LASER', 'LATER', 'LAUGH', 'LAYER', 'LEARN', 'LEASE', 'LEAST', 'LEAVE', 'LEGAL', 'LEVEL', 'LIGHT', 'LIMIT', 'LINKS', 'LIVES', 'LOCAL', 'LOOSE', 'LOWER', 'LUCKY', 'LUNCH', 'LYING', 'MAGIC', 'MAJOR', 'MAKER', 'MARCH', 'MATCH', 'MAYBE', 'MAYOR', 'MEANT', 'MEDIA', 'METAL', 'MIGHT', 'MINOR', 'MINUS', 'MIXED', 'MODEL', 'MONEY', 'MONTH', 'MORAL', 'MOTOR', 'MOUNT', 'MOUSE', 'MOUTH', 'MOVED', 'MOVIE', 'MUSIC', 'NEEDS', 'NEVER', 'NEWLY', 'NIGHT', 'NOISE', 'NORTH', 'NOTED', 'NOVEL', 'NURSE', 'OCCUR', 'OCEAN', 'OFFER', 'OFTEN', 'ORDER', 'OTHER', 'OUGHT', 'PAINT', 'PANEL', 'PAPER', 'PARTY', 'PEACE', 'PHASE', 'PHONE', 'PHOTO', 'PIANO', 'PIECE', 'PILOT', 'PITCH', 'PLACE', 'PLAIN', 'PLANE', 'PLANT', 'PLATE', 'POINT', 'POUND', 'POWER', 'PRESS', 'PRICE', 'PRIDE', 'PRIME', 'PRINT', 'PRIOR', 'PRIZE', 'PROOF', 'PROUD', 'PROVE', 'QUEEN', 'QUICK', 'QUIET', 'QUITE', 'RADIO', 'RAISE', 'RANGE', 'RAPID', 'RATIO', 'REACH', 'READY', 'REALM', 'REBEL', 'REFER', 'RELAX', 'RELAY', 'REPLY', 'RIGHT', 'RIGID', 'RIVAL', 'RIVER', 'ROBOT', 'ROMAN', 'ROUGH', 'ROUND', 'ROUTE', 'ROYAL', 'RURAL', 'SCALE', 'SCENE', 'SCOPE', 'SCORE', 'SENSE', 'SERVE', 'SEVEN', 'SHALL', 'SHAPE', 'SHARE', 'SHARP', 'SHEET', 'SHELF', 'SHELL', 'SHIFT', 'SHINE', 'SHIRT', 'SHOCK', 'SHOOT', 'SHORT', 'SHOWN', 'SIDES', 'SIGHT', 'SILLY', 'SINCE', 'SIXTH', 'SIXTY', 'SIZED', 'SKILL', 'SLEEP', 'SLIDE', 'SMALL', 'SMART', 'SMILE', 'SMITH', 'SMOKE', 'SOLID', 'SOLVE', 'SORRY', 'SOUND', 'SOUTH', 'SPACE', 'SPARE', 'SPEAK', 'SPEED', 'SPEND', 'SPENT', 'SPLIT', 'SPOKE', 'SPORT', 'STAFF', 'STAGE', 'STAKE', 'STAND', 'START', 'STATE', 'STEAM', 'STEEL', 'STEEP', 'STEER', 'STICK', 'STILL', 'STOCK', 'STONE', 'STOOD', 'STORE', 'STORM', 'STORY', 'STRIP', 'STUCK', 'STUDY', 'STUFF', 'STYLE', 'SUGAR', 'SUITE', 'SUPER', 'SWEET', 'SWIFT', 'SWING', 'SWISS', 'TABLE', 'TAKEN', 'TASTE', 'TAXES', 'TEACH', 'TEENS', 'TEETH', 'TEMPO', 'TERMS', 'TEXAS', 'THANK', 'THEFT', 'THEIR', 'THEME', 'THERE', 'THESE', 'THICK', 'THING', 'THINK', 'THIRD', 'THOSE', 'THREE', 'THREW', 'THROW', 'THUMB', 'TIGER', 'TIGHT', 'TIMER', 'TIRED', 'TITLE', 'TODAY', 'TOPIC', 'TOTAL', 'TOUCH', 'TOUGH', 'TOWER', 'TRACK', 'TRADE', 'TRAIL', 'TRAIN', 'TREAT', 'TREND', 'TRIAL', 'TRIBE', 'TRICK', 'TRIED', 'TRIES', 'TRUCK', 'TRULY', 'TRUNK', 'TRUST', 'TRUTH', 'TWICE', 'TWIST', 'ULTRA', 'UNCLE', 'UNDER', 'UNDUE', 'UNION', 'UNITY', 'UNTIL', 'UPPER', 'UPSET', 'URBAN', 'USAGE', 'USUAL', 'VALID', 'VALUE', 'VIDEO', 'VIRUS', 'VISIT', 'VITAL', 'VOCAL', 'VOICE', 'WASTE', 'WATCH', 'WATER', 'WHEEL', 'WHERE', 'WHICH', 'WHILE', 'WHITE', 'WHOLE', 'WHOSE', 'WOMAN', 'WOMEN', 'WORLD', 'WORRY', 'WORSE', 'WORST', 'WORTH', 'WOULD', 'WRITE', 'WRONG', 'WROTE', 'YOUNG', 'YOUTH'
        }
    
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
        """Adaptive letter detection that handles multiple puzzle layouts"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            h, w = img.shape[:2]
            
            # Try multiple detection strategies in order of reliability
            
            # Strategy 1: Look for circular letter wheel
            letters = self._detect_circular_wheel(img)
            if letters and len(letters) >= 3:
                logger.info(f"Found {len(letters)} letters using circular detection")
                return letters
            
            # Strategy 2: Look for grid-based letters in bottom half
            letters = self._detect_grid_layout(img)
            if letters and len(letters) >= 3:
                logger.info(f"Found {len(letters)} letters using grid detection")
                return letters
            
            # Strategy 3: General contour-based detection
            letters = self._detect_letters_fallback(img)
            if letters and len(letters) >= 3:
                logger.info(f"Found {len(letters)} letters using fallback detection")
                return letters
                
            # Strategy 4: Search entire image for any text
            letters = self._detect_letters_full_scan(img)
            logger.info(f"Found {len(letters)} letters using full scan")
            return letters
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _detect_circular_wheel(self, img):
        """Detect letters arranged in a circular wheel pattern"""
        h, w = img.shape[:2]
        
        # Search in bottom 50% of image for circular patterns
        search_start_y = int(h * 0.5)
        search_region = img[search_start_y:h, :]
        
        # Convert to grayscale for circle detection
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        
        # Try multiple circle detection parameters
        circle_params = [
            (50, 30, 60, 150),   # Standard detection
            (40, 25, 50, 200),   # More sensitive
            (60, 35, 80, 180),   # Less sensitive
        ]
        
        for param1, param2, minR, maxR in circle_params:
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                                     param1=param1, param2=param2, 
                                     minRadius=minR, maxRadius=maxR)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for circle in sorted(circles, key=lambda c: c[2], reverse=True):
                    cx, cy, radius = circle
                    
                    # Adjust coordinates to full image
                    center_x = cx
                    center_y = cy + search_start_y
                    
                    # Extract the circular region
                    padding = 30
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
                    if letters and len(letters) >= 3:
                        return letters
        
        return []
    
    def _detect_grid_layout(self, img):
        """Detect letters arranged in a grid layout"""
        h, w = img.shape[:2]
        
        # Focus on bottom half where letter grids are typically located
        grid_region_y = int(h * 0.4)
        grid_region = img[grid_region_y:h, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(grid_region, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold to handle varying lighting
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 15, 8)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for letter-sized rectangles
        letter_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200 or area > 8000:  # Reasonable letter size range
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            # Check aspect ratio (should be roughly square for letters)
            aspect_ratio = w_box / h_box
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
            
            # Check minimum size
            if w_box < 20 or h_box < 20:
                continue
            
            letter_candidates.append((x, y, w_box, h_box, area))
        
        # Sort by area and process largest candidates first
        letter_candidates.sort(key=lambda x: x[4], reverse=True)
        
        letters = []
        for x, y, w_box, h_box, area in letter_candidates[:10]:  # Limit to top 10
            letter_region = grid_region[y:y+h_box, x:x+w_box]
            letter = self._ocr_letter(letter_region)
            
            if letter and letter.isalpha():
                center_x_abs = x + w_box // 2
                center_y_abs = grid_region_y + y + h_box // 2
                
                letters.append({
                    'letter': letter.upper(),
                    'position': [center_x_abs, center_y_abs]
                })
        
        return letters
    
    def _detect_letters_full_scan(self, img):
        """Full image scan for any detectable letters"""
        h, w = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try different threshold methods
        threshold_methods = [
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU),
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2),
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
        ]
        
        all_letters = []
        
        for _, binary in threshold_methods:
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
                area = cv2.contourArea(contour)
                if area < 100 or area > 10000:
                    continue
                
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Basic size and aspect ratio checks
                if w_box < 15 or h_box < 15 or w_box > 150 or h_box > 150:
                    continue
                
                aspect_ratio = w_box / h_box
                if aspect_ratio < 0.1 or aspect_ratio > 10.0:
                    continue
                
                letter_region = img[y:y+h_box, x:x+w_box]
                letter = self._ocr_letter(letter_region)
                
                if letter and letter.isalpha():
                    center_x_abs = x + w_box // 2
                    center_y_abs = y + h_box // 2
                    
                    # Avoid duplicates
                    is_duplicate = False
                    for existing in all_letters:
                        if (abs(existing['position'][0] - center_x_abs) < 30 and 
                            abs(existing['position'][1] - center_y_abs) < 30):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        all_letters.append({
                            'letter': letter.upper(),
                            'position': [center_x_abs, center_y_abs]
                        })
        
        return all_letters[:8]  # Return up to 8 letters
    
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
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
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