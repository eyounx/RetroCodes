import pygame
import cv2
import threading
from pygame.locals import *
from retro_contest.local import make
import sys
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

import argparse

train_level = [['SonicTheHedgehog-Genesis', 'SpringYardZone.Act3'],
			   ['SonicTheHedgehog-Genesis', 'SpringYardZone.Act2'],
			   ['SonicTheHedgehog-Genesis', 'GreenHillZone.Act3'],
			   ['SonicTheHedgehog-Genesis', 'GreenHillZone.Act1'],
			   ['SonicTheHedgehog-Genesis', 'StarLightZone.Act2'],
			   ['SonicTheHedgehog-Genesis', 'StarLightZone.Act1'],
			   ['SonicTheHedgehog-Genesis', 'MarbleZone.Act2'],
			   ['SonicTheHedgehog-Genesis', 'MarbleZone.Act1'],
			   ['SonicTheHedgehog-Genesis', 'MarbleZone.Act3'],
			   ['SonicTheHedgehog-Genesis', 'ScrapBrainZone.Act2'],
			   ['SonicTheHedgehog-Genesis', 'LabyrinthZone.Act2'],
			   ['SonicTheHedgehog-Genesis', 'LabyrinthZone.Act1'],
			   ['SonicTheHedgehog-Genesis', 'LabyrinthZone.Act3'],
			   ['SonicTheHedgehog2-Genesis', 'EmeraldHillZone.Act1'],
			   ['SonicTheHedgehog2-Genesis', 'EmeraldHillZone.Act2'],
			   ['SonicTheHedgehog2-Genesis', 'ChemicalPlantZone.Act2'],
			   ['SonicTheHedgehog2-Genesis', 'ChemicalPlantZone.Act1'],
			   ['SonicTheHedgehog2-Genesis', 'MetropolisZone.Act1'],
			   ['SonicTheHedgehog2-Genesis', 'MetropolisZone.Act2'],
			   ['SonicTheHedgehog2-Genesis', 'OilOceanZone.Act1'],
			   ['SonicTheHedgehog2-Genesis', 'OilOceanZone.Act2'],
			   ['SonicTheHedgehog2-Genesis', 'MysticCaveZone.Act2'],
			   ['SonicTheHedgehog2-Genesis', 'MysticCaveZone.Act1'],
			   ['SonicTheHedgehog2-Genesis', 'HillTopZone.Act1'],
			   ['SonicTheHedgehog2-Genesis', 'CasinoNightZone.Act1'],
			   ['SonicTheHedgehog2-Genesis', 'WingFortressZone'],
			   ['SonicTheHedgehog2-Genesis', 'AquaticRuinZone.Act2'],
			   ['SonicTheHedgehog2-Genesis', 'AquaticRuinZone.Act1'],
			   ['SonicAndKnuckles3-Genesis', 'LavaReefZone.Act2'],
			   ['SonicAndKnuckles3-Genesis', 'CarnivalNightZone.Act2'],
			   ['SonicAndKnuckles3-Genesis', 'CarnivalNightZone.Act1'],
			   ['SonicAndKnuckles3-Genesis', 'MarbleGardenZone.Act1'],
			   ['SonicAndKnuckles3-Genesis', 'MarbleGardenZone.Act2'],
			   ['SonicAndKnuckles3-Genesis', 'MushroomHillZone.Act2'],
			   ['SonicAndKnuckles3-Genesis', 'MushroomHillZone.Act1'],
			   ['SonicAndKnuckles3-Genesis', 'DeathEggZone.Act1'],
			   ['SonicAndKnuckles3-Genesis', 'DeathEggZone.Act2'],
			   ['SonicAndKnuckles3-Genesis', 'FlyingBatteryZone.Act1'],
			   ['SonicAndKnuckles3-Genesis', 'SandopolisZone.Act1'],
			   ['SonicAndKnuckles3-Genesis', 'SandopolisZone.Act2'],
			   ['SonicAndKnuckles3-Genesis', 'HiddenPalaceZone'],
			   ['SonicAndKnuckles3-Genesis', 'HydrocityZone.Act2'],
			   ['SonicAndKnuckles3-Genesis', 'IcecapZone.Act1'],
			   ['SonicAndKnuckles3-Genesis', 'IcecapZone.Act2'],
			   ['SonicAndKnuckles3-Genesis', 'AngelIslandZone.Act1'],
			   ['SonicAndKnuckles3-Genesis', 'LaunchBaseZone.Act2'],
			   ['SonicAndKnuckles3-Genesis', 'LaunchBaseZone.Act1']]

level_num = len(train_level)
save_interval = 10
nn = 0;

#	newObs[112 - D : 112 + D, 160 - D : 160 + D, :] = 0
#	lastObs[112 - D : 112 + D, 160 - D : 160 + D, :] = 0

C = 4
D = 50
def refine(kp, des):
	nkp = [];
	nde = [];
	for i in range(len(kp)):
		#print(kp[i].pt[0], kp[i].pt[1])
		if not ((160 - D) / C <= kp[i].pt[0] and kp[i].pt[0] <= (160 + D) / C):
			nkp.append(kp[i])
			nde.append(des[i])

	return np.array(nkp), np.array(nde)

def restart():
	# rg = int(random.random() * len(train_level))
	global current_level
	rg = (current_level + 10) % level_num
	env = make(game=train_level[rg][0], state=train_level[rg][1])

#    print(train_level[rg][0], train_level[rg][1])
	return env, train_level[rg][0], train_level[rg][1]

def check(x, y):
	return 0 <= x and x < h and 0 <= y and y < w;

def same(a, b):
	#print(a)
	#print(b)
	for i in range(len(a)):
		#print(a[i], b[i], a[i] - b[i])
		if abs(int(a[i]) - int(b[i])) > 6:
			return False;
	return True


def smooth(x):
	for y in range(w):
		all = 0;
		dif = 0;
		for dx in range(-3, 3):
			for dy in range(-3, 3):
				if check(x + dx, y + dy):
					all = all + 1;
					if mask[x][y] != mask[x + dx][y + dy]:
						dif += 1;
		if dif > all * 0.7:
			bmask[x][y] = 1
		else:
			bmask[x][y] = 0;

class myThread (threading.Thread):
    def __init__(self, x):
        threading.Thread.__init__(self)
        self.x = x
    def run(self):
        smooth(self.x)

def main(path):
	bs = cv2.createBackgroundSubtractorKNN(detectShadows = False)
	count = 0
	# pygame.init()
	w = 320
	h = 224
	bmask = [[0 for y in range(w)] for x in range(h)]
	#mask = [[0 for y in range(w)] for x in range(h)]
	rgbdiff = np.zeros((224,320), dtype=np.uint8)
	lastmask = np.zeros((224,320), dtype=np.uint8)

	lastImage = np.zeros((224,320,3), dtype=np.uint8)
	voting = np.zeros((224,320,3), dtype=np.uint8)
	lastBg = np.zeros((224,320,3), dtype=np.uint8)
	bgimg = np.zeros((224,320,3), dtype=np.uint8)
	lastObs = np.zeros((224,320,3), dtype=np.uint8)
	newObs = np.zeros((224,320,3), dtype=np.uint8)

	bg = (213, 1, 144)
	#bg = (213, 1, 144)
	size = width, height = 320, 224
	size2 = 660, 458

	screen = pygame.display.set_mode(size2, RESIZABLE)
	# screen = pygame.display.set_mode((width, height))
	pygame.display.set_caption("debug sonic game")
	# pygame.display.update()
	# while True:
	#     for event in pygame.event.get():
	#         if event.type == pygame.QUIT:
	#             pygame.quit()
	#             quit()

	env, game, level = restart()
	obs = env.reset()
	allc = {};
	frame_i = 0
	game_j = 0
	state = obs
	fgbg = cv2.createBackgroundSubtractorMOG2()
	xx = 96
	yy = 108
	while True:
		pygame.event.pump()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()

		action = np.zeros(12)
		pressed_keys = pygame.key.get_pressed()

		if pressed_keys[K_LEFT]:
			action[6] = 1
		if pressed_keys[K_RIGHT]:
			action[7] = 1
		if pressed_keys[K_DOWN]:
			action[5] = 1
		if pressed_keys[K_UP]:
			action[1] = 1
		if pressed_keys[K_b]:
			action[1] = 1

		obs, rew, done, info = env.step(action)
		# print (obs.shape)

		if done:
			env.reset()

		img = pygame.image.frombuffer(obs.tobytes(), (320,224), "RGB")

		frame_i += 1
		#print(imgFile, frame_i)
		if frame_i > 10000:
			exit()

		state = obs

		screen.fill(bg)
		screen.blit(img, (0,0))
		bgimg = bgimg.astype(np.uint8)
		#img2 = pygame.image.frombuffer(bgimg, (320,224), "RGB")
		opt = fgbg.apply(obs)
		#cv2.imshow('frame',opt)

		opt = opt.repeat(3)
		opt = 255 - opt
		opt = np.reshape(opt, (h, w, 3))
		opt = opt.astype(np.uint8)

	#	img2 = pygame.image.frombuffer(opt, (320,224), "RGB")
	#	screen.blit(img2, (0,224 + 10))

		img3 = pygame.image.frombuffer(bgimg, (320,224), "RGB")
		screen.blit(img3, (325,0))

		#blur = cv2.medianBlur(bgimg,3)
		#cv2.imshow("Bilateral", blur)

		truemask = np.zeros((224,320), dtype=np.uint8)
		'''	
		for d in range(3):
			for i in lastImage[200:220:5, 160 - (d - 1) * 5]:
				
				if np.sum(i) <= 60:
					continue;
				mask3 = (obs - i)
				mask3[mask3 <= 5] = 1;
				mask3[mask3 > 250] = 1;
				mask3[mask3 > 1] = 0;
				mask = np.sum(mask3, axis = 2)
				mask[mask < 3] = 0
				mask[mask == 3] = 1;
				if np.sum(mask) * 0.618 < np.sum(mask[120:, :]):
					truemask += mask
		'''
		truemask += np.multiply(lastmask, 1 - rgbdiff)
		truemask[truemask > 1] = 1
		lastmask[:, :] = truemask[:, :]
		bgmask = truemask.repeat(3) * 255
		bgimg3 = np.reshape(bgmask, (h, w, 3))
		bgimg3 = bgimg3.astype(np.uint8)

		#sift = cv2.SIFT()
		#print(obs.shape)
		cl = np.sum(obs, axis=(0,1))
		nc = (cl[:] / 100000);
		#if (int(nc[0]), int(nc[1]), int(nc[2])) not in allc:
		#	print((int(nc[0]), int(nc[1]), int(nc[2])))
		#	allc[(int(nc[0]), int(nc[1]), int(nc[2]))] = 1;
		if frame_i > 1:
			surf =  cv2.xfeatures2d.SURF_create()

			#sift = cv2.xfeatures2d.SIFT_create()
			ss = (int(320 / C), int(224 / C))
			#ss2 = (int(640 / C), int(224 / C))
			newObs[:, :, :] = obs[:, :, :]
			kp1, des1 = surf.detectAndCompute(cv2.resize(newObs, ss), None)
			kp2, des2 = surf.detectAndCompute(cv2.resize(lastObs, ss), None)
		#	print(des1)
			newObs[112 - D : 112 + D, 160 - D : 160 + D, :] = 0
			lastObs[112 - D : 112 + D, 160 - D : 160 + D, :] = 0

			kp1, des1 = refine(kp1, des1)
		#	print(des1)
			kp2, des2 = refine(kp2, des2)
			bf = cv2.BFMatcher()
			matches = bf.knnMatch(des1, des2, k=2)
			good = []
			dx = 0
			dy = 0

			try:
				for m, n in matches:
					if m.distance < 0.75 * n.distance:
						good.append([m])
					#	print(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt);
					#	dx += kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0];
					#	dy += kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1];
			except None:
				print("no enough kp")
			status = np.ones((4, 1, 1), dtype=np.uint8)

			realgood = [];
			if len(good) > 4:
				ptsA= np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
				ptsB = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
				ransacReprojThreshold = 4
				H, status = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
			upb = int(len(good) / 2 - (len(status) - sum(sum(status))));
			#print(upb)
			for limit in range(upb):
				best = -1
				for i in range(len(good)):
					if status[i] != [1]:
						continue
					if best == -1:
						best = i;
					elif good[i][0].distance > good[best][0].distance:
						best = i
			#	print(best)
				if best != -1:
					status[best] = [0];
			for i in range(len(good)):
				if status[i] == [1]:
					realgood.append(good[i]);
					dx += kp1[good[i][0].queryIdx].pt[0] - kp2[good[i][0].trainIdx].pt[0];
					dy += kp1[good[i][0].queryIdx].pt[1] - kp2[good[i][0].trainIdx].pt[1];
			for it in kp1:
				it.pt = (it.pt[0] * C, it.pt[1] * C)
			for it in kp2:
				it.pt = (it.pt[0] * C, it.pt[1] * C)
			imgx = cv2.drawMatchesKnn(newObs, kp1, lastObs, kp2, realgood, None, flags=2)
			#imgx = cv2.drawMatchesKnn(cv2.resize(obs, ss), kp1, cv2.resize(lastObs, ss), kp2, good, None, flags=2)
			img2 = pygame.image.frombuffer(imgx, (640, 224), "RGB")
			screen.blit(img2, (0,224 + 10))
			if len(realgood):
				xx -= 5.348484848484849 * dx / len(realgood)
				yy -= 4.5 * dy / len(realgood)
			print(int(xx), int(yy), info['x'], info['y']);



		#print(bgimg3[:, :, :])
		#print(bgimg3.shape)
		#blur = cv2.medianBlur(bgimg3,5)
		#cv2.imshow("Bilateral", bgimg3)



		#opt2 = bs.apply(obs)

		##opt2 = opt2.repeat(3)
		#opt2 = 255 - opt2
		#opt2 = np.reshape(opt2, (h, w, 3))
		#opt2 = opt2.astype(np.uint8)
		#img4 = pygame.image.frombuffer(opt2, (320,224), "RGB")
		#screen.blit(img4, (325,234))


		lastObs = obs;
		pygame.display.flip()
		# pygame.display.update()

		pygame.time.delay(30)

if __name__ == '__main__':
	'''
	如果是anaconda安装，会出现无法接受接受键盘输入
	解决：
		用pythonw命令取代python命令。
		如果目录下没有pythonw文件，则执行
		conda install python.app 
	
	按键
		s : start recording
		d : end   recording
	
		left
		right
		up
		down
		b=up
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument("--level", type=int, default=1)
	parser.add_argument("--save_path", type=str, default="./replay")
	args = parser.parse_args()

	global current_level
	current_level = args.level
	path = ""
	main(path)
