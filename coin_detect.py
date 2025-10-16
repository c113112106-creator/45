import cv2
import numpy as np

#讀取圖片
img = cv2.imread("coins.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 3)

#偵測邊緣
edges = cv2.Canny(blur, 50, 150)

#找圓形
circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=60,
    param1=100,
    param2=30,
    minRadius=20,
    maxRadius=100
)

#若偵測到圓
coin_count = 0
coin_stats = {"1": 0, "5": 0, "10": 0, "50": 0}

if circles is not None:
    circles = np.uint16(np.around(circles))
    coin_count = len(circles[0, :])

    for i in circles[0, :]:
        x, y, r = i
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)  #畫外圈
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)  #畫中心點

        # 根據半徑分類）
        if r < 52:
            coin_type = "1"
        elif 52 <= r < 58:
            coin_type = "5"
        elif 58 <= r < 70:
            coin_type = "10"
        else:
            coin_type = "50"

        coin_stats[coin_type] += 1

        #標示種類文字
        cv2.putText(img, coin_type, (x - 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#顯示總數量
cv2.putText(img, f"Total: {coin_count}", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#顯示各類別統計
y_offset = 90
for coin, num in coin_stats.items():
    cv2.putText(img, f"{coin}: {num}", (30, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y_offset += 40

#輸出成品圖
cv2.imwrite("coins_result.jpg", img)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
