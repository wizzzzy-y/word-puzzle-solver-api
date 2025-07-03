import os
import logging
import requests
import time
import itertools
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract

# Try to import OpenCV, but use PIL fallback if it fails
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError as e:
    logging.warning(f"OpenCV not available: {e}. Using PIL fallback.")
    OPENCV_AVAILABLE = False

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
            'ABOUT', 'ABOVE', 'ABUSE', 'ACTOR', 'ACUTE', 'ADMIT', 'ADOPT', 'ADULT', 'AFTER', 'AGAIN', 'AGENT', 'AGREE', 'AHEAD', 'ALARM', 'ALBUM', 'ALERT', 'ALIEN', 'ALIGN', 'ALIKE', 'ALIVE', 'ALLOW', 'ALONE', 'ALONG', 'ALTER', 'ANGLE', 'ANGRY', 'APART', 'APPLE', 'APPLY', 'ARENA', 'ARGUE', 'ARISE', 'ARRAY', 'ASIDE', 'ASSET', 'AVOID', 'AWAKE', 'AWARD', 'AWARE', 'BADLY', 'BASIC', 'BEACH', 'BEGAN', 'BEGIN', 'BEING', 'BENCH', 'BIRTH', 'BLACK', 'BLAME', 'BLANK', 'BLIND', 'BLOCK', 'BLOOD', 'BOARD', 'BOOST', 'BOOTH', 'BOUND', 'BRAIN', 'BRAND', 'BREAD', 'BREAK', 'BREED', 'BRIEF', 'BRING', 'BROAD', 'BROKE', 'BROWN', 'BUILD', 'BUILT', 'CATCH', 'CAUSE', 'CHAIN', 'CHAIR', 'CHAOS', 'CHARM', 'CHART', 'CHASE', 'CHEAP', 'CHECK', 'CHEST', 'CHIEF', 'CHILD', 'CHINA', 'CHOSE', 'CIVIL', 'CLAIM', 'CLASS', 'CLEAN', 'CLEAR', 'CLICK', 'CLIMB', 'CLOCK', 'CLOSE', 'CLOUD', 'COACH', 'COAST', 'COULD', 'COUNT', 'COURT', 'COVER', 'CRAFT', 'CRASH', 'CRAZY', 'CREAM', 'CRIME', 'CROSS', 'CROWD', 'CROWN', 'CRUDE', 'CURVE', 'CYCLE', 'DAILY', 'DANCE', 'DATED', 'DEALT', 'DEATH', 'DEBUT', 'DELAY', 'DEPTH', 'DOING', 'DOUBT', 'DOZEN', 'DRAFT', 'DRAMA', 'DRANK', 'DREAM', 'DRESS', 'DRILL', 'DRINK', 'DRIVE', 'DROVE', 'DYING', 'EAGER', 'EARLY', 'EARTH', 'EIGHT', 'ELITE', 'EMPTY', 'ENEMY', 'ENJOY', 'ENTER', 'ENTRY', 'EQUAL', 'ERROR', 'EVENT', 'EVERY', 'EXACT', 'EXIST', 'EXTRA', 'FAITH', 'FALSE', 'FAULT', 'FIELD', 'FIFTH', 'FIFTY', 'FIGHT', 'FINAL', 'FIRST', 'FIXED', 'FLASH', 'FLEET', 'FLOOR', 'FLUID', 'FOCUS', 'FORCE', 'FORTH', 'FORTY', 'FORUM', 'FOUND', 'FRAME', 'FRANK', 'FRAUD', 'FRESH', 'FRONT', 'FRUIT', 'FULLY', 'FUNNY', 'GIANT', 'GIVEN', 'GLASS', 'GLOBE', 'GOING', 'GRACE', 'GRADE', 'GRAND', 'GRANT', 'GRASS', 'GRAVE', 'GREAT', 'GREEN', 'GROSS', 'GROUP', 'GROWN', 'GUARD', 'GUESS', 'GUEST', 'GUIDE', 'HAPPY', 'HARSH', 'HEART', 'HEAVY', 'HENCE', 'HORSE', 'HOTEL', 'HOUSE', 'HUMAN', 'IDEAL', 'IMAGE', 'INDEX', 'INNER', 'INPUT', 'ISSUE', 'JAPAN', 'JOINT', 'JUDGE', 'KNOWN', 'LABEL', 'LARGE', 'LASER', 'LATER', 'LAUGH', 'LAYER', 'LEARN', 'LEASE', 'LEAST', 'LEAVE', 'LEGAL', 'LEVEL', 'LIGHT', 'LIMIT', 'LINKS', 'LIVES', 'LOCAL', 'LOOSE', 'LOWER', 'LUCKY', 'LUNCH', 'LYING', 'MAGIC', 'MAJOR', 'MAKER', 'MARCH', 'MATCH', 'MAYBE', 'MAYOR', 'MEANT', 'MEDIA', 'METAL', 'MIGHT', 'MINOR', 'MINUS', 'MIXED', 'MODEL', 'MONEY', 'MONTH', 'MORAL', 'MOTOR', 'MOUNT', 'MOUSE', 'MOUTH', 'MOVED', 'MOVIE', 'MUSIC', 'NEEDS', 'NEVER', 'NEWLY', 'NIGHT', 'NOISE', 'NORTH', 'NOTED', 'NOVEL', 'NURSE', 'OCCUR', 'OCEAN', 'OFFER', 'OFTEN', 'ORDER', 'OTHER', 'OUGHT', 'PAINT', 'PANEL', 'PAPER', 'PARTY', 'PEACE', 'PHASE', 'PHONE', 'PHOTO', 'PIANO', 'PIECE', 'PILOT', 'PITCH', 'PLACE', 'PLAIN', 'PLANE', 'PLANT', 'PLATE', 'POINT', 'POUND', 'POWER', 'PRESS', 'PRICE', 'PRIDE', 'PRIME', 'PRINT', 'PRIOR', 'PRIZE', 'PROOF', 'PROUD', 'PROVE', 'QUEEN', 'QUICK', 'QUIET', 'QUITE', 'RADIO', 'RAISE', 'RANGE', 'RAPID', 'RATIO', 'REACH', 'READY', 'REALM', 'REBEL', 'REFER', 'RELAX', 'RELAY', 'REPLY', 'RIGHT', 'RIGID', 'RIVAL', 'RIVER', 'ROBOT', 'ROMAN', 'ROUGH', 'ROUND', 'ROUTE', 'ROYAL', 'RURAL', 'SCALE', 'SCENE', 'SCOPE', 'SCORE', 'SENSE', 'SERVE', 'SEVEN', 'SHALL', 'SHAPE', 'SHARE', 'SHARP', 'SHEET', 'SHELF', 'SHELL', 'SHIFT', 'SHINE', 'SHIRT', 'SHOCK', 'SHOOT', 'SHORT', 'SHOWN', 'SIDES', 'SIGHT', 'SILLY', 'SINCE', 'SIXTH', 'SIXTY', 'SIZED', 'SKILL', 'SLEEP', 'SLIDE', 'SMALL', 'SMART', 'SMILE', 'SMITH', 'SMOKE', 'SOLID', 'SOLVE', 'SORRY', 'SOUND', 'SOUTH', 'SPACE', 'SPARE', 'SPEAK', 'SPEED', 'SPEND', 'SPENT', 'SPLIT', 'SPOKE', 'SPORT', 'STAFF', 'STAGE', 'STAKE', 'STAND', 'START', 'STATE', 'STEAM', 'STEEL', 'STEEP', 'STEER', 'STICK', 'STILL', 'STOCK', 'STONE', 'STOOD', 'STORE', 'STORM', 'STORY', 'STRIP', 'STUCK', 'STUDY', 'STUFF', 'STYLE', 'SUGAR', 'SUITE', 'SUPER', 'SWEET', 'SWIFT', 'SWING', 'SWISS', 'TABLE', 'TAKEN', 'TASTE', 'TAXES', 'TEACH', 'TEENS', 'TEETH', 'TEMPO', 'TERMS', 'TEXAS', 'THANK', 'THEFT', 'THEIR', 'THEME', 'THERE', 'THESE', 'THICK', 'THING', 'THINK', 'THIRD', 'THOSE', 'THREE', 'THREW', 'THROW', 'THUMB', 'TIGER', 'TIGHT', 'TIMER', 'TIRED', 'TITLE', 'TODAY', 'TOPIC', 'TOTAL', 'TOUCH', 'TOUGH', 'TOWER', 'TRACK', 'TRADE', 'TRAIL', 'TRAIN', 'TREAT', 'TREND', 'TRIAL', 'TRIBE', 'TRICK', 'TRIED', 'TRIES', 'TRUCK', 'TRULY', 'TRUNK', 'TRUST', 'TRUTH', 'TWICE', 'TWIST', 'ULTRA', 'UNCLE', 'UNDER', 'UNDUE', 'UNION', 'UNITY', 'UNTIL', 'UPPER', 'UPSET', 'URBAN', 'USAGE', 'USUAL', 'VALID', 'VALUE', 'VIDEO', 'VIRUS', 'VISIT', 'VITAL', 'VOCAL', 'VOICE', 'WASTE', 'WATCH', 'WATER', 'WHEEL', 'WHERE', 'WHICH', 'WHILE', 'WHITE', 'WHOLE', 'WHOSE', 'WOMAN', 'WOMEN', 'WORLD', 'WORRY', 'WORSE', 'WORST', 'WORTH', 'WOULD', 'WRITE', 'WRONG', 'WROTE', 'YOUNG', 'YOUTH',
            # Additional useful words for word games
            'PART', 'TRAP', 'RAPT', 'TARP', 'PRAT', 'ART', 'RAT', 'PAT', 'TAR', 'TAP'
        }

    def solve_puzzle(self, image_path):
        """Main puzzle solving function"""
        try:
            letters_data = self._detect_letters(image_path)
            if not letters_data:
                logger.warning("No letters detected in image")
                return {"swipes": []}

            logger.info(f"Detected {len(letters_data)} letters")
            swipes = self._generate_word_swipes(letters_data)
            return {"swipes": swipes}

        except Exception as e:
            logger.error(f"Solver error: {e}")
            return {"swipes": []}

    def _detect_letters(self, image_path):
        """Enhanced letter detection with PIL fallback when OpenCV fails"""
        try:
            if OPENCV_AVAILABLE:
                return self._detect_letters_opencv(image_path)
            else:
                return self._detect_letters_pil(image_path)

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def _detect_letters_opencv(self, image_path):
        """OpenCV-based letter detection"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error("Failed to load image")
                return []

            h, w = img.shape[:2]
            logger.info(f"Processing image dimensions: {w}x{h}")

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
            logger.error(f"OpenCV detection error: {e}")
            return []

    def _detect_letters_pil(self, image_path):
        """PIL-based letter detection as fallback"""
        try:
            img = Image.open(image_path)
            w, h = img.size
            logger.info(f"Processing image dimensions: {w}x{h} (PIL mode)")

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Strategy 1: Detect letters in bottom circular area
            letters = self._detect_circular_wheel_pil(img)
            if letters and len(letters) >= 3:
                logger.info(f"Found {len(letters)} letters using PIL circular detection")
                return letters

            # Strategy 2: Try direct OCR on different sections
            letters = self._detect_letters_sections_pil(img)
            logger.info(f"Found {len(letters)} letters using PIL sectional detection")
            return letters

        except Exception as e:
            logger.error(f"PIL detection error: {e}")
            return []

    def _detect_circular_wheel_pil(self, img):
        """Detect letters in circular arrangement using PIL"""
        try:
            w, h = img.size
            
            # Focus on bottom portion where circular wheel typically is
            bottom_y = int(h * 0.6)
            bottom_section = img.crop((0, bottom_y, w, h))
            
            # Convert to grayscale and enhance
            gray = bottom_section.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray)
            enhanced = enhancer.enhance(2.0)
            
            # Apply threshold to get binary image
            threshold = 128
            binary = enhanced.point(lambda x: 255 if x > threshold else 0, '1')
            
            # Use OCR on the processed bottom section
            text = pytesseract.image_to_string(binary, config=self.tesseract_config).strip()
            
            letters = []
            if text:
                # Extract individual letters and estimate positions
                unique_letters = list(set(c.upper() for c in text if c.isalpha()))
                for i, letter in enumerate(unique_letters):
                    # Estimate positions in a circular arrangement
                    angle = (i * 2 * 3.14159) / len(unique_letters)
                    radius = min(w, h - bottom_y) // 4
                    center_x = w // 2
                    center_y = (h - bottom_y) // 2
                    
                    x = int(center_x + radius * 0.7 * (1 if i % 2 == 0 else -1))
                    y = int(center_y + radius * 0.7 * (1 if i < len(unique_letters)//2 else -1))
                    
                    letters.append({
                        'letter': letter,
                        'x': x,
                        'y': y + bottom_y,
                        'confidence': 0.7
                    })
            
            return letters
            
        except Exception as e:
            logger.error(f"PIL circular detection error: {e}")
            return []

    def _detect_letters_sections_pil(self, img):
        """Detect letters by sectioning the image with PIL"""
        try:
            w, h = img.size
            letters = []
            
            # Try different sections of the image
            sections = [
                (0, int(h * 0.6), w, h),  # Bottom section
                (0, 0, w, int(h * 0.4)),  # Top section  
                (0, int(h * 0.3), w, int(h * 0.7)),  # Middle section
            ]
            
            for i, (x1, y1, x2, y2) in enumerate(sections):
                try:
                    section = img.crop((x1, y1, x2, y2))
                    
                    # Convert to grayscale and enhance
                    gray = section.convert('L')
                    enhancer = ImageEnhance.Contrast(gray)
                    enhanced = enhancer.enhance(1.5)
                    
                    # Use OCR to detect text
                    text = pytesseract.image_to_string(enhanced, config=self.tesseract_config).strip()
                    
                    if text:
                        unique_letters = list(set(c.upper() for c in text if c.isalpha()))
                        for j, letter in enumerate(unique_letters):
                            letters.append({
                                'letter': letter,
                                'x': x1 + (x2 - x1) // 2 + j * 20,
                                'y': y1 + (y2 - y1) // 2,
                                'confidence': 0.6
                            })
                            
                except Exception as section_error:
                    logger.warning(f"Section {i} processing failed: {section_error}")
                    continue
            
            # Remove duplicates while preserving order
            seen = set()
            unique_letters = []
            for letter_data in letters:
                letter = letter_data['letter']
                if letter not in seen:
                    seen.add(letter)
                    unique_letters.append(letter_data)
            
            return unique_letters
            
        except Exception as e:
            logger.error(f"PIL sectional detection error: {e}")
            return []

    def _detect_circular_wheel(self, img):
        """Detect letters in circular arrangement (like the bottom of your image)"""
        try:
            h, w = img.shape[:2]
            
            # Focus on bottom portion where circular wheel typically is
            bottom_section = img[int(h * 0.6):, :]
            
            # Convert to grayscale and apply preprocessing
            gray = cv2.cvtColor(bottom_section, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold for better text detection
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find circles using HoughCircles
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=10, maxRadius=100)
            
            letters = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # Extract region around circle
                    roi = thresh[max(0, y-r):y+r, max(0, x-r):x+r]
                    if roi.size > 0:
                        # Use OCR to detect letter
                        letter = pytesseract.image_to_string(roi, config=self.tesseract_config).strip()
                        if letter and letter.isalpha() and len(letter) == 1:
                            letters.append({
                                'letter': letter.upper(),
                                'x': x,
                                'y': y + int(h * 0.6),  # Adjust for bottom section offset
                                'confidence': 0.8
                            })
            
            return letters
            
        except Exception as e:
            logger.error(f"Circular detection error: {e}")
            return []

    def _detect_grid_layout(self, img):
        """Detect letters in grid layout"""
        try:
            h, w = img.shape[:2]
            
            # Focus on middle-bottom area where grids typically are
            grid_section = img[int(h * 0.2):int(h * 0.8), :]
            
            gray = cv2.cvtColor(grid_section, cv2.COLOR_BGR2GRAY)
            
            # Apply morphological operations to isolate text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 15, 4)
            
            # Find contours with OpenCV compatibility fix
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            
            letters = []
            for contour in contours:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Filter for reasonable letter-sized rectangles
                if 15 <= w_rect <= 60 and 15 <= h_rect <= 60:
                    # Extract ROI
                    roi = thresh[y:y+h_rect, x:x+w_rect]
                    
                    # Use OCR
                    letter = pytesseract.image_to_string(roi, config=self.tesseract_config).strip()
                    if letter and letter.isalpha() and len(letter) == 1:
                        letters.append({
                            'letter': letter.upper(),
                            'x': x + w_rect // 2,
                            'y': y + h_rect // 2 + int(h * 0.2),  # Adjust for section offset
                            'confidence': 0.7
                        })
            
            return letters
            
        except Exception as e:
            logger.error(f"Grid detection error: {e}")
            return []

    def _detect_letters_fallback(self, img):
        """General contour-based letter detection with OpenCV compatibility"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use Otsu's thresholding
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours with compatibility fix
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            
            letters = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 5000:  # Filter by area
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Filter for letter-like shapes
                    if 0.2 < aspect_ratio < 2.0:
                        # Extract ROI and pad it
                        roi = thresh[y:y+h, x:x+w]
                        
                        # Pad the ROI for better OCR
                        padded_roi = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
                        
                        # Use OCR
                        letter = pytesseract.image_to_string(padded_roi, config=self.tesseract_config).strip()
                        if letter and letter.isalpha() and len(letter) == 1:
                            letters.append({
                                'letter': letter.upper(),
                                'x': x + w // 2,
                                'y': y + h // 2,
                                'confidence': 0.6
                            })
            
            return letters
            
        except Exception as e:
            logger.error(f"Fallback detection error: {e}")
            return []

    def _detect_letters_full_scan(self, img):
        """Full image scan for any text - last resort"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Try different preprocessing approaches
            approaches = [
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            ]
            
            all_letters = []
            
            for thresh in approaches:
                # Use OCR on entire image
                try:
                    data = pytesseract.image_to_data(thresh, config='--psm 6', output_type=pytesseract.Output.DICT)
                    
                    for i, text in enumerate(data['text']):
                        if text.strip() and text.isalpha() and len(text.strip()) == 1:
                            conf = int(data['conf'][i])
                            if conf > 30:  # Minimum confidence threshold
                                x = data['left'][i] + data['width'][i] // 2
                                y = data['top'][i] + data['height'][i] // 2
                                
                                all_letters.append({
                                    'letter': text.strip().upper(),
                                    'x': x,
                                    'y': y,
                                    'confidence': conf / 100.0
                                })
                except Exception:
                    continue
            
            # Remove duplicates (letters detected multiple times)
            unique_letters = []
            for letter in all_letters:
                is_duplicate = False
                for existing in unique_letters:
                    if (letter['letter'] == existing['letter'] and 
                        abs(letter['x'] - existing['x']) < 30 and 
                        abs(letter['y'] - existing['y']) < 30):
                        is_duplicate = True
                        # Keep the one with higher confidence
                        if letter['confidence'] > existing['confidence']:
                            unique_letters.remove(existing)
                            unique_letters.append(letter)
                        break
                
                if not is_duplicate:
                    unique_letters.append(letter)
            
            return unique_letters
            
        except Exception as e:
            logger.error(f"Full scan error: {e}")
            return []

    def _generate_word_swipes(self, letters_data):
        """Generate valid word combinations and their swipe paths"""
        try:
            if not letters_data:
                return []
            
            available_letters = [letter['letter'] for letter in letters_data]
            logger.info(f"Available letters: {available_letters}")
            
            valid_words = []
            
            # Try different word lengths
            for length in range(3, min(len(available_letters) + 1, 8)):
                for combination in itertools.permutations(available_letters, length):
                    word = ''.join(combination)
                    if word in self.dictionary:
                        # Calculate swipe path
                        path = self._calculate_swipe_path(word, letters_data)
                        if path:
                            valid_words.append({
                                'word': word,
                                'path': path,
                                'score': len(word) * 10  # Simple scoring
                            })
            
            # Sort by score and word length
            valid_words.sort(key=lambda x: (-x['score'], -len(x['word'])))
            
            # Return top 20 words to avoid overwhelming response
            return valid_words[:20]
            
        except Exception as e:
            logger.error(f"Word generation error: {e}")
            return []

    def _calculate_swipe_path(self, word, letters_data):
        """Calculate the swipe path for a given word"""
        try:
            path = []
            used_positions = set()
            
            for letter in word:
                # Find available position for this letter
                found = False
                for i, letter_data in enumerate(letters_data):
                    if (letter_data['letter'] == letter and i not in used_positions):
                        path.append({
                            'x': letter_data['x'],
                            'y': letter_data['y'],
                            'letter': letter
                        })
                        used_positions.add(i)
                        found = True
                        break
                
                if not found:
                    # Letter not available, invalid word
                    return None
            
            return path
            
        except Exception as e:
            logger.error(f"Path calculation error: {e}")
            return None

# Backward compatibility function
def solve_word_puzzle(image_path):
    """Backward compatibility wrapper"""
    solver = WordPuzzleSolver()
    return solver.solve_puzzle(image_path)
