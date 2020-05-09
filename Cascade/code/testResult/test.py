import cv2

path = 'test.xml'
shipCascade = cv2.CascadeClassifier(path)

flag = 0
trueD = 0
falseD = 0
partD = 0
while True:
    flag += 1
    picName = 'test/test_' + str(flag) + '.jpg'
    img = cv2.imread(picName,cv2.IMREAD_COLOR)
    cv2.imshow('img', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rect = shipCascade.detectMultiScale(gray, 1.2, 8)
    for (x, y, w, h) in rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('img', img)
    k = cv2.waitKey(0) & 0xFF  # 等待按键
    if k == ord('1'):
        trueD += 1
        cv2.destroyAllWindows()
    elif k == ord('3'):
        falseD += 1
        cv2.destroyAllWindows()
    elif k == ord('2'):
        partD += 1
        cv2.destroyAllWindows()
    elif k == 27:   #	wait for ESC
        cv2.destroyAllWindows()
        break
    if flag == 1000:
        break
f = open('result.txt', 'w')
f.write('True :'+ str(trueD) + '\nFalse :' + str(falseD) + '\nPart :' + str(partD))
f.close()