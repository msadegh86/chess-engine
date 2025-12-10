# chess_paper_detector_color_full.py
import cv2
import numpy as np
import time

# ==================== تنظیمات ====================
# اگر از IP-Webcam استفاده می‌کنی اینجا آدرس را بذار (مثال: "http://192.168.1.5:8080/video")
# اگر می‌خوای از وب‌کم لپ‌تاپ استفاده کنی بذار url = 0
url = "http://10.220.90.254:8080/video"

BOARD_SIZE = 800
ROWS, COLS = 8, 8
letters = ['a','b','c','d','e','f','g','h']

# پارامترهای baseline / تشخیص تغییر
MEAN_DIFF_THRESH = 18        # آستانه اختلاف میانگین خاکستری
CHANGED_RATIO_THRESH = 0.03  # نسبت پیکسل‌های تغییر کرده در هر خانه
WHITE_RATIO_THRESH = 0.06    # نسبت سفید جدید کمک‌کننده
DIFF_PIXEL_THRESHOLD = 25    # آستانه دیفرانسیل پیکسل برای diff_img

# پارامترهای رنگ (HSV)
# آبی
BLUE_LOWER = np.array([90, 70, 50])
BLUE_UPPER = np.array([130, 255, 255])
# قرمز (دو بازه)
RED1_LOWER = np.array([0, 90, 50])
RED1_UPPER = np.array([10, 255, 255])
RED2_LOWER = np.array([160, 90, 50])
RED2_UPPER = np.array([180, 255, 255])

# نسبت حداقل رنگ در خانه برای تشخیص "موجود بودن برگه رنگی"
COLOR_RATIO_THRESH = 0.04   # 4% از سطح خانه --> بسته به سایز برگه تنظیم کن

KERNEL = np.ones((3,3), np.uint8)

# مجموعه‌های ذخیره برای خانه‌های دیده‌شده
visited_blue_cells = set()
visited_red_cells = set()

# لیست آخرین وضعیت برگه‌ها روی برد (بر اساس آخرین فریم معتبر)
latest_blue_squares = []   # مثل ["a3", "c5", ...]
latest_red_squares  = []

# حداقل نسبت مساحت برای اینکه یک کانتور را به‌عنوان برگه در نظر بگیریم
MIN_PAPER_AREA_RATIO = 0.01  # حدود ۱٪ از یک خانه (در صورت نیاز تنظیم کن)

# ================ توابع کمکی ================
def open_capture(url):
    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            # تلاش دوم: اگر url قابل تبدیل به int باشه (مثلاً 0) امتحان کن
            try:
                cap = cv2.VideoCapture(int(url))
            except:
                pass
        return cap
    except Exception as e:
        print("open_capture error:", e)
        return None

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect

def detect_papers_positions(board_img):
    """
    ورودی: تصویر warp شده‌ی کل صفحه‌ی ۸×۸
    خروجی:
        blue_squares: لیست نام خانه‌هایی که برگه آبی در آن است (مرتب بر اساس y سپس x)
        red_squares:  لیست نام خانه‌هایی که برگه قرمز در آن است (مرتب بر اساس y سپس x)
    """
    h, w = board_img.shape[:2]
    cell_h = h // ROWS
    cell_w = w // COLS

    hsv = cv2.cvtColor(board_img, cv2.COLOR_BGR2HSV)

    # ماسک آبی
    mask_blue = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, KERNEL)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, KERNEL)

    # ماسک قرمز (دو بازه)
    mask_red1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
    mask_red2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
    mask_red  = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red  = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, KERNEL)
    mask_red  = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, KERNEL)

    min_area = cell_w * cell_h * MIN_PAPER_AREA_RATIO

    def extract_squares_from_mask(mask):
        # پیدا کردن کانتورهای برگه‌ها روی ماسک
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []  # (cy, cx, name)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            col = int(cx // cell_w)
            row = int(cy // cell_h)

            if col < 0 or col >= COLS or row < 0 or row >= ROWS:
                continue

            name = f"{letters[col]}{8-row}"  # تبدیل مختصات به مثل a3, b5, ...

            # برای مرتب‌سازی: اول بر اساس y (cy) بعد x (cx)
            candidates.append((cy, cx, name))

        # مرتب‌سازی از بالا به پایین، و در صورت برابر بودن y، از چپ به راست
        candidates.sort(key=lambda x: (x[0], x[1]))

        # فقط حداکثر ۵ تا را نگه می‌داریم
        squares = [item[2] for item in candidates[:5]]
        return squares

    blue_squares = extract_squares_from_mask(mask_blue)
    red_squares  = extract_squares_from_mask(mask_red)

    return blue_squares, red_squares

# برای انتخاب دستی 4 نقطه (manual mode)
manual_pts = []
def mouse_callback(event, x, y, flags, param):
    global manual_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        manual_pts.append((x,y))
        print("manual click:", x, y)

# ==================== شروع برنامه ====================
cap = open_capture(url)
if cap is None or not cap.isOpened():
    print("❌ دوربین باز نشد. آدرس را چک کن یا از وب‌کم محلی استفاده کن (url=0).")
    exit()

print("شروع برنامه. کلیدها: b=baseline, m=manual select, d=debug, q/ESC=exit")

cv2.namedWindow("ManualSelect")
cv2.setMouseCallback("ManualSelect", mouse_callback)

baseline_cells_mean = None
baseline_cells_image = None
baseline_taken = False

prev_detected = None
debug_info = None

# loop اصلی
while True:
    ret, frame = cap.read()
    if not ret:
        print("فریم دریافت نشد — اتصال یا آدرس را بررسی کن.")
        time.sleep(0.5)
        continue

    # یک نسخه کوچک برای نمایش سریع
    display = cv2.resize(frame, (640,480))

    # ------------- پیدا کردن کانتور بزرگ (صفحه) -------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # شرط چهارگوش و حداقل مساحت برای گرفتن صفحه
            if len(approx) == 4 and area > 10000:
                page_contour = approx
                max_area = area

    warped = None
    debug_info = []

    # ------------- اگر اتوماتیک پیدا نشد و manual_pts هست از آن استفاده کن -------------
    if page_contour is None and len(manual_pts) == 4:
        pts = np.array(manual_pts, dtype="float32")
        h0, w0 = frame.shape[:2]
        # نقاطی که با موس گرفتیم در پنجره‌ی 640x480 هستند؛ به ابعاد اصلی map کنیم
        sx = w0 / 640.0
        sy = h0 / 480.0
        pts_scaled = np.array([(int(x*sx), int(y*sy)) for (x,y) in manual_pts], dtype="float32")
        rect = order_points(pts_scaled)
        dst = np.array([[0,0],[BOARD_SIZE,0],[BOARD_SIZE,BOARD_SIZE],[0,BOARD_SIZE]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (BOARD_SIZE, BOARD_SIZE))

    elif page_contour is not None:
        pts = page_contour.reshape(4,2)
        rect = order_points(pts)
        dst = np.array([[0,0],[BOARD_SIZE,0],[BOARD_SIZE,BOARD_SIZE],[0,BOARD_SIZE]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (BOARD_SIZE, BOARD_SIZE))

    detected_cell = None
    detected_colors_in_frame = []  # برای نمایش سریع چه چیزهایی در این فریم پیدا شد

    # ------------- اگر warped ساخته شد (صفحه صاف شده) -------------
    if warped is not None:
        cell_h = BOARD_SIZE // ROWS
        cell_w = BOARD_SIZE // COLS

        # اگر baseline گرفته شده → مقایسه کن
        if baseline_taken and baseline_cells_mean is not None:
            for r in range(ROWS):
                for c in range(COLS):
                    x = c * cell_w
                    y = r * cell_h
                    cell = warped[y:y+cell_h, x:x+cell_w]

                    # میانگین خاکستری فعلی و مقایسه با baseline
                    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                    mean_now = np.mean(gray_cell)
                    mean_base = baseline_cells_mean[r][c]
                    diff_mean = abs(mean_now - mean_base)

                    # مقایسه تصویر پیکسلی (diff)
                    base_img = baseline_cells_image[r][c]
                    if base_img.shape != gray_cell.shape:
                        base_img = cv2.resize(base_img, (gray_cell.shape[1], gray_cell.shape[0]))
                    diff_img = cv2.absdiff(gray_cell, base_img)
                    _, diff_th = cv2.threshold(diff_img, DIFF_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)
                    diff_ratio = cv2.countNonZero(diff_th) / (cell_w * cell_h)

                    # تشخیص سفید اضافه (کمکی)
                    hsv_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
                    mask_white = cv2.inRange(hsv_cell, np.array([0,0,200]), np.array([179,60,255]))
                    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, KERNEL)
                    white_ratio = cv2.countNonZero(mask_white) / (cell_w * cell_h)

                    debug_info.append(((r,c), mean_base, mean_now, diff_mean, diff_ratio, white_ratio))

                    # تصمیم نهایی تغییر
                    changed = False
                    if (diff_mean > MEAN_DIFF_THRESH and diff_ratio > CHANGED_RATIO_THRESH) or \
                       (white_ratio > WHITE_RATIO_THRESH and diff_ratio > 0.01):
                        changed = True

                    # اگر خانه تغییر داشته باشه، بررسی رنگ (همزمان آبی و قرمز)
                    name = f"{letters[c]}{8-r}"
                    color_found_here = []

                    if changed:
                        # ماسک آبی و قرمز روی cell
                        mask_blue = cv2.inRange(hsv_cell, BLUE_LOWER, BLUE_UPPER)
                        mask_red1 = cv2.inRange(hsv_cell, RED1_LOWER, RED1_UPPER)
                        mask_red2 = cv2.inRange(hsv_cell, RED2_LOWER, RED2_UPPER)
                        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

                        # مورفولوژی برای حذف نویز
                        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, KERNEL)
                        mask_red  = cv2.morphologyEx(mask_red,  cv2.MORPH_OPEN, KERNEL)

                        blue_ratio = cv2.countNonZero(mask_blue) / (cell_w * cell_h)
                        red_ratio  = cv2.countNonZero(mask_red)  / (cell_w * cell_h)

                        # اگر نسبت کافی بود خانه را علامت بزن و ذخیره کن
                        if blue_ratio > COLOR_RATIO_THRESH:
                            visited_blue_cells.add(name)
                            color_found_here.append("blue")
                        if red_ratio > COLOR_RATIO_THRESH:
                            visited_red_cells.add(name)
                            color_found_here.append("red")

                        # اگر هر رنگی پیدا شد، برای نمایش سریع در فریم نشان بده
                        if color_found_here:
                            detected_cell = (r,c)
                            detected_colors_in_frame.append((name, color_found_here))

                    # همیشه نام خانه را بنویس (برای کاربر)
                    cv2.putText(warped, name, (x+6, y+26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)
                    cv2.rectangle(warped, (x,y), (x+cell_w, y+cell_h), (0,255,0), 1)

        else:
            # هنوز baseline نداریم — فقط نام خانه‌ها و خطوط رو نشون بده
            for r in range(ROWS):
                for c in range(COLS):
                    x = c * cell_w
                    y = r * cell_h
                    name = f"{letters[c]}{8-r}"
                    cv2.putText(warped, name, (x+6, y+26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)
                    cv2.rectangle(warped, (x,y), (x+cell_w, y+cell_h), (0,255,0), 1)

        # اگر در این فریم خانه‌ای با رنگ پیدا شده، نمایش نمونه و متن
        if detected_cell is not None:
            # می‌توان چندتا رنگ همزمان داشته باشیم؛ نمایش آخرین یافته برای simplicity
            r,c = detected_cell
            name = f"{letters[c]}{8-r}"
            # متن خلاصه رنگ‌های پیدا شده در آخرین خانه
            text = []
            for nm, colors in detected_colors_in_frame:
                # nm همان مثل اسم خونه است؛ باید متن کلی را بسازیم
                txt = nm + ":" + ",".join(colors)
                text.append(txt)
            # نمایش متن‌ها در پایین صفحه
            bottom_text = " | ".join(text)
            cv2.putText(warped, bottom_text, (10, BOARD_SIZE-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # نمایش نمونه‌ی اولین خانه‌ی پیدا شده
            sample = warped[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            cv2.imshow("Paper Sample", cv2.resize(sample, (220,220)))
        else:
            # اگر هیچ خانه‌ای پیدا نشده پنجره نمونه را ببند
            if cv2.getWindowProperty("Paper Sample", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Paper Sample")

        # --- تشخیص موقعیت نهایی برگه‌های آبی و قرمز بر اساس آخرین فریم warped ---
        latest_blue_squares, latest_red_squares = detect_papers_positions(warped)

    # ------------- نمایش پنجره‌ها -------------
    cv2.imshow("Camera Live", display)
    if warped is not None:
        cv2.imshow("Warped Board", cv2.resize(warped, (600,600)))
    cv2.imshow("ManualSelect", cv2.resize(display, (640,480)))

    key = cv2.waitKey(1) & 0xFF

    # کلیدها
    if key == 27 or key == ord('q'):
        break

    elif key == ord('b'):
        # گرفتن baseline (حتمی: قبل از زدن b هیچ برگه‌ای روی خانه‌ها نباشد)
        if warped is None:
            print("Baseline: صفحه مشخص نشده، ابتدا صفحه را در فریم قرار بده.")
            continue
        print("Baseline taken. Make sure board has NO paper on it when you press 'b'.")
        baseline_cells_mean = [[0]*COLS for _ in range(ROWS)]
        baseline_cells_image = [[None]*COLS for _ in range(ROWS)]
        cell_h = BOARD_SIZE // ROWS
        cell_w = BOARD_SIZE // COLS
        for r in range(ROWS):
            for c in range(COLS):
                x = c * cell_w
                y = r * cell_h
                cell = warped[y:y+cell_h, x:x+cell_w]
                gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                baseline_cells_mean[r][c] = np.mean(gray_cell)
                baseline_cells_image[r][c] = gray_cell.copy()
        baseline_taken = True

    elif key == ord('m'):
        # وارد حالت دستی شو؛ کاربر 4 کلیک انجام می‌دهد
        print("Manual mode: روی پنجره 'ManualSelect' ۴ گوشه را کلیک کن (TL, TR, BR, BL) سپس Enter را بزن.")
        manual_pts = []
        while True:
            cv2.imshow("ManualSelect", cv2.resize(display, (640,480)))
            k = cv2.waitKey(0) & 0xFF
            if k == 13 or k == 10:  # Enter
                if len(manual_pts) == 4:
                    print("۴ نقطه ثبت شد:", manual_pts)
                else:
                    print("نقاط کمتر از ۴ هستند:", len(manual_pts))
                break
            elif k == 27:
                print("انصراف از manual")
                break

    elif key == ord('d'):
        # چاپ debug نمونه ها
        if debug_info:
            print("Sample debug (r,c, mean_base, mean_now, diff_mean, diff_ratio, white_ratio):")
            for item in debug_info[:16]:
                print(item)
        else:
            print("No debug info yet.")

# پایان حلقه — cleanup و چاپ نتایج نهایی
cap.release()
cv2.destroyAllWindows()

print("\n==================== نتایج نهایی ====================")
print("🔵 خانه‌های آبی (visited):", sorted(list(visited_blue_cells)))
print("🔴 خانه‌های قرمز (visited):", sorted(list(visited_red_cells)))

# ساخت دیکشنری نهایی برای برگه‌ها (حداکثر ۵ عدد)
def build_paper_dict(squares):
    result = {}
    for i in range(5):
        if i < len(squares):
            result[i+1] = squares[i]
        else:
            result[i+1] = None
    return result

blue_papers = build_paper_dict(latest_blue_squares)
red_papers  = build_paper_dict(latest_red_squares)

print("\n🔵 blue_papers (وضعیت نهایی برگه‌های آبی):")
print(blue_papers)

print("\n🔴 red_papers (وضعیت نهایی برگه‌های قرمز):")
print(red_papers)
print("====================================================\n")
