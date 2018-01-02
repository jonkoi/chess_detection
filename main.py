# Starting code from
# coding=utf-8
import PIL.Image
import matplotlib.image as mpimg
import scipy.ndimage
import cv2 # For Sobel etc
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
np.set_printoptions(suppress=True, linewidth=200) # Better printing of arrays
plt.rcParams['image.cmap'] = 'jet' # Default colormap is jet
import scipy as sp
from scipy.stats import norm
from scipy.misc import imsave
from detection import detection_img
import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
from scipy.misc import imsave
import os

###########################################################################
### From starting code (few code changes but little change from source) ###
###########################################################################
def getMinSaddleDist(saddle_pts, pt):
    best_dist = None
    best_pt = pt
    for saddle_pt in saddle_pts:
        saddle_pt = saddle_pt[::-1]
        dist = np.sum((saddle_pt - pt)**2)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pt = saddle_pt
    return best_pt, np.sqrt(best_dist)

def is_square(cnt, eps=3.0, xratio_thresh = 0.5):
  center = cnt.sum(axis=0)/4

  dd0 = np.sqrt(((cnt[0,:] - cnt[1,:])**2).sum())
  dd1 = np.sqrt(((cnt[1,:] - cnt[2,:])**2).sum())
  dd2 = np.sqrt(((cnt[2,:] - cnt[3,:])**2).sum())
  dd3 = np.sqrt(((cnt[3,:] - cnt[0,:])**2).sum())

  xa = np.sqrt(((cnt[0,:] - cnt[2,:])**2).sum())
  xb = np.sqrt(((cnt[1,:] - cnt[3,:])**2).sum())
  xratio = xa/xb if xa < xb else xb/xa

  ta = getAngle(dd3, dd0, xb)
  tb = getAngle(dd0, dd1, xa)
  tc = getAngle(dd1, dd2, xb)
  td = getAngle(dd2, dd3, xa)
  angle_sum = np.round(ta+tb+tc+td)

  is_convex = np.abs(angle_sum - 360) < 5

  angles = np.array([ta,tb,tc,td])
  good_angles = np.all((angles > 40) & (angles < 140))

  dda = dd0 / dd1
  if dda < 1:
    dda = 1. / dda
  ddb = dd1 / dd2
  if ddb < 1:
    ddb = 1. / ddb
  ddc = dd2 / dd3
  if ddc < 1:
    ddc = 1. / ddc
  ddd = dd3 / dd0
  if ddd < 1:
    ddd = 1. / ddd
  side_ratios = np.array([dda,ddb,ddc,ddd])
  good_side_ratios = np.all(side_ratios < eps)

  return (good_angles)

def getAngle(a,b,c):
  k = (a*a+b*b-c*c) / (2*a*b)
  if (k < -1):
    k=-1
  elif k > 1:
    k=1
  return np.arccos(k) * 180.0 / np.pi

def pruneContours(contours, hierarchy, saddle):
  new_contours = []
  new_hierarchies = []
  for i in range(len(contours)):
    cnt = contours[i]
    h = hierarchy[i]
    if h[2] != -1:
        continue
    if len(cnt) != 4:
      continue
    if cv2.contourArea(cnt) < 8*8:
      continue
    if not is_square(cnt):
      continue
    cnt = updateCorners(cnt, saddle)
    if len(cnt) != 4:
        continue

    new_contours.append(cnt)
    new_hierarchies.append(h)

  new_contours = np.array(new_contours)
  new_hierarchy = np.array(new_hierarchies)
  return np.array(new_contours), np.array(new_hierarchy)

def getContours(img, edges):
    # Morphological Gradient to get internal squares of canny edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    _, contours, hierarchy = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
      # Approximate contour and update in place
      contours[i] = cv2.approxPolyDP(contours[i],0.04*cv2.arcLength(contours[i],True),True)

    return np.array(contours), hierarchy[0]

def updateCorners(contour, saddle):
    ws = 4 # half window size (+1)
    new_contour = contour.copy()
    for i in range(len(contour)):
        cc,rr = contour[i,0,:]
        rl = max(0,rr-ws)
        cl = max(0,cc-ws)
        window = saddle[rl:min(saddle.shape[0],rr+ws+1),cl:min(saddle.shape[1],cc+ws+1)]
        br, bc = np.unravel_index(window.argmax(), window.shape)
        s_score = window[br,bc]
        br -= min(ws,rl)
        bc -= min(ws,cl)
        if s_score > 0:
            new_contour[i,0,:] = cc+bc,rr+br
        else:
            return []
    return new_contour

def getIdentityGrid(N):
    a = np.arange(N)
    b = a.copy()
    aa,bb = np.meshgrid(a,b)
    return np.vstack([aa.flatten(), bb.flatten()]).T

def findGoodPoints(grid, spts, max_px_dist=5):
    new_grid = grid.copy()
    chosen_spts = set()
    N = len(new_grid)
    grid_good = np.zeros(N,dtype=np.bool)
    hash_pt = lambda pt: "%d_%d" % (pt[0], pt[1])

    for pt_i in range(N):
        pt2, d = getMinSaddleDist(spts, grid[pt_i,:2].A.flatten())
        if hash_pt(pt2) in chosen_spts:
            d = max_px_dist
        else:
            chosen_spts.add(hash_pt(pt2))
        if (d < max_px_dist): # max dist to replace with
            new_grid[pt_i,:2] = pt2
            grid_good[pt_i] = True
    return new_grid, grid_good

def getInitChessGrid(quad):
    quadA = np.array([[0,1],[1,1],[1,0],[0,0]],dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))
    return makeChessGrid(M,1)

def makeChessGrid(M, N=1):
    ideal_grid = getIdentityGrid(2+2*N)-N
    ideal_grid_pad = np.pad(ideal_grid, ((0,0),(0,1)), 'constant', constant_values=1) # Add 1's column
    # warped_pts = M*pts
    grid = (np.matrix(M)*ideal_grid_pad.T).T
    grid[:,:2] /= grid[:,2] # normalize by t
    grid = grid[:,:2] # remove 3rd column
    return grid, ideal_grid, M

def generateNewBestFit(grid_ideal, grid, grid_good):
    a = np.float32(grid_ideal[grid_good])
    b = np.float32(grid[grid_good])
    M = cv2.findHomography(a, b, cv2.RANSAC)
    return M

def getGrads(img):
    img = cv2.blur(img,(5,5))
    gx = cv2.Sobel(img,cv2.CV_64F,1,0)
    gy = cv2.Sobel(img,cv2.CV_64F,0,1)

    grad_mag = gx*gx+gy*gy
    grad_phase = np.arctan2(gy, gx) # from -pi to pi
    grad_phase_masked = grad_phase.copy()
    gradient_mask_threshold = 2*np.mean(grad_mag.flatten())
    grad_phase_masked[grad_mag < gradient_mask_threshold] = np.nan
    return grad_mag, grad_phase_masked, grad_phase, gx, gy


def getBestLines(img_warped):
    grad_mag, grad_phase_masked, grad_phase, gx, gy = getGrads(img_warped)

    # X
    gx_pos = gx.copy()
    gx_pos[gx_pos < 0] = 0
    gx_neg = -gx.copy()
    gx_neg[gx_neg < 0] = 0
    score_x = np.sum(gx_pos, axis=0) * np.sum(gx_neg, axis=0)
    # Y
    gy_pos = gy.copy()
    gy_pos[gy_pos < 0] = 0
    gy_neg = -gy.copy()
    gy_neg[gy_neg < 0] = 0
    score_y = np.sum(gy_pos, axis=1) * np.sum(gy_neg, axis=1)

    # Choose best internal set of 7
    a = np.array([(offset + np.arange(7) + 1)*32 for offset in np.arange(1,11-2)])
    scores_x = np.array([np.sum(score_x[pts]) for pts in a])
    scores_y = np.array([np.sum(score_y[pts]) for pts in a])

    # 15x15 grid, so along an axis a set of 7, and an internal 7 at that, so 13x13 grid, 7x7 possibility inside
    # We're also using a 1-padded grid so 17x17 grid
    # We only want the internal choices (13-7) so 6x6 possible options in the 13x13
    # so 2,3,4,5,6,7,8 to 8,9,10,11,12,13,14 ignoring 0,1 and 15,16,17
    best_lines_x = a[scores_x.argmax()]
    best_lines_y = a[scores_y.argmax()]
    return (best_lines_x, best_lines_y)


def GetContours(image):
    edges = cv2.Canny(image, 20, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # edges_gradient = cv2.dilate(edges, kernel, iterations = 1)
    edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    _, contours, hierarchy = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        contours[i] = cv2.approxPolyDP(contours[i],0.04*cv2.arcLength(contours[i],True),True)
    return new_contours

def findChessboard(img, min_pts_needed=15, max_pts_needed=10):
    gray = cv2.blur(img, (3,3))
    saddle, spts, d = getSaddle(gray)
    edges = cv2.Canny(img, 20, 250)
    contours_all, hierarchy = getContours(img, edges)
    contours, hierarchy = pruneContours(contours_all, hierarchy, saddle)
    # print(contours.shape)
    # print(contours_all.shape)

    curr_num_good = 0
    curr_grid_next = None
    curr_grid_good = None
    curr_M = None

    for cnt_i in range(len(contours)):
        cnt = contours[cnt_i].squeeze()
        grid_curr, ideal_grid, M = getInitChessGrid(cnt)

        for grid_i in range(7):
            grid_curr, ideal_grid, _ = makeChessGrid(M, N=(grid_i+1))
            grid_next, grid_good = findGoodPoints(grid_curr, spts)
            num_good = np.sum(grid_good)
            if num_good < 4:
                M = None
                break
            M, _ = generateNewBestFit(ideal_grid, grid_next, grid_good)
            if M is None or np.abs(M[0,0] / M[1,1]) > 15 or np.abs(M[1,1] / M[0,0]) > 15:
                M = None
                break
        if M is None:
            continue
        elif num_good > curr_num_good:
            curr_num_good = num_good
            curr_grid_next = grid_next
            curr_grid_good = grid_good
            curr_M = M
        if num_good > max_pts_needed:
            break
    if curr_num_good > min_pts_needed:
        final_ideal_grid = getIdentityGrid(2+2*7)-7
        return curr_M, final_ideal_grid, curr_grid_next, curr_grid_good, spts,d, saddle
    else:
        # Rollback to old saddle point detection without candidate elimination

        return None, None, None, None, None, d, saddle

#####################################################################################
### End: From starting code (few code changes but little change from source code) ###
#####################################################################################

###############################################################################################################################
### Written code: Saddle point detection and elimination from improved Hessian matrix, and integration with piece detection ###
###############################################################################################################################
def loadImage(filepath, grayCvt=True):
    image = PIL.Image.open(filepath);
    w, h = image.size

    w_percent = 500/w;
    new_h = int(h*w_percent);
    resized_image = image.resize((500,new_h), PIL.Image.ANTIALIAS)
    if grayCvt:
        resized_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_BGR2GRAY)
    return resized_image

def sharpenImage(img):
    kernel = np.zeros((9,9), np.float32)
    kernel[4,4] = 2.0
    boxFilter = np.ones((9,9), np.float32)/81.0
    kernel = kernel - boxFilter
    output = cv2.filter2D(img, -1, kernel)
    return output

def getSaddle(gray_img):
    img = gray_img.astype(np.float64)
    gx = cv2.Sobel(img,cv2.CV_64F,1,0)
    gy = cv2.Sobel(img,cv2.CV_64F,0,1)
    gxx = cv2.Sobel(gx,cv2.CV_64F,1,0)
    gyy = cv2.Sobel(gy,cv2.CV_64F,0,1)
    gxy = cv2.Sobel(gx,cv2.CV_64F,0,1)
    l1_map = 0.5*(gxx + gyy + np.sqrt( (gxx-gyy)**2 + 4*gxy**2))
    l2_map = 0.5*(gxx + gyy - np.sqrt( (gxx-gyy)**2 + 4*gxy**2))
    epsilon = 0.2*l1_map.max()

    # First criterion
    S = gxx*gyy - gxy**2
    S[S > 0] = 255
    S[l1_map < epsilon] = 255
    S[l2_map > -epsilon] = 255
    S = -S

    #Local extreme
    win = 10
    w, h = img.shape
    S2 = np.zeros_like(S, dtype=np.float64)
    for i,j in np.argwhere(S):
        # Get neigborhood
        ta=max(0,i-win)
        tb=min(w,i+win+1)
        tc=max(0,j-win)
        td=min(h,j+win+1)
        cell = S[ta:tb,tc:td]
        val = S[i,j]

        if cell.max() == val:
            S2[i,j] = val
    S2[S2 < 255] = 0
    S2[S2 > 254] = 255

    #Find parameter
    spts = np.argwhere(S2)
    spts_tree = sp.spatial.cKDTree(spts)

    closest_3s = spts_tree.query(spts, k=4)
    closest_dist = closest_3s[0][:,1].astype(int)
    counts = np.bincount(closest_dist)
    peak = np.argmax(counts)

    max_win = min(peak, len(counts) - peak-1)
    gauss_counts = peak
    j = 0
    for i in range(max_win):
        gauss_counts = np.sum(counts[peak - i: peak + i])
        j = i
        if gauss_counts > int(0.8*len(spts)):
            break
    mean,std=norm.fit(counts[peak-j:peak+j])
    mean = peak

    aMin = mean - 3*std
    aMax = mean + 3*std

    #TUNE THIS!
    r = 0.5*closest_dist[closest_dist.argmin()]
    p = 0.5*aMax/aMin
    d = 4*aMax
    t = 0.95 # 15 degree
    print('t is: ', t)

    print('Spts 0 : ', len(spts))
    a = np.array([0,7,6,5,4,3,2,1])
    angles = np.pi/4 * a

    #Second criterion
    delete_idx = []
    for idx, pt in enumerate(spts):
        pt = [pt[1],pt[0]]
        pts = np.zeros((8,2))
        for i in range(8):
            pt_x = pt[0]
            pt_y = pt[1]
            x = int(min(max(pt_x + r*np.cos(angles[i]),0), h))
            y = int(min(max(pt_y + r*np.sin(angles[i]),0), w))
            pts[i] = np.array([x,y])
        I = np.zeros(8)
        for i in range(8):
            tri = np.float32([[pt,  pts[i], pts[(i+1)%8]]])
            mask = np.zeros((w,h),dtype = np.uint8)
            cv2.fillPoly(mask, np.int32(tri), 255)
            temp_img =np.bitwise_and(gray_img, mask).astype(np.uint8)
            I[i] = np.sum(temp_img)
            # print(i, I[i])
            # cv2.imshow('Imagee', S2)
            # cv2.imshow('Image', temp_img)
            # cv2.waitKey(0)
        D1 = abs(I[0] - I[4])
        D2 = abs(I[2] - I[6])
        D3 = abs(I[0] + I[4] - I[2] - I[6]) / 2
        D4 = abs(I[1] - I[5])
        D5 = abs(I[3] - I[7])
        D6 = abs(I[1] + I[5] - I[3] - I[7]) / 2

        if not (((D1 < p*D3 and D2 < p*D3) or (D4 < p*D6 and D5 < p*D6))):
            delete_idx.append(idx)
    spts1 = np.delete(spts, delete_idx, axis = 0)
    # spts1 = spts
    if (len(spts1) < 15):
        print('rollback')
        spts1 = spts
    print (len(spts1))

    #Third criterion
    spts1_tree = sp.spatial.cKDTree(spts1)
    closest_3s_1 = spts1_tree.query(spts1, k=4)
    delete_idx = []
    for idx, pt in enumerate(spts1):
        max_distance =closest_3s_1[0][idx, 3]
        if max_distance > d:
            # print('distance')
            delete_idx.append(idx)
            continue
        first = spts1[closest_3s_1[1][idx, 1]]
        second = spts1[closest_3s_1[1][idx, 2]]
        v1 = first - pt
        v2 = second - pt
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if cosine_angle > t:
            # print(cosine_angle)
            delete_idx.append(idx)
    spts2 = np.delete(spts1, delete_idx, axis = 0)
    # spts2 =spts1

    #Create new saddle image
    new_S = np.zeros((w,h), dtype=np.uint8)
    for pt in spts2:
        new_S[pt[0], pt[1]] = 255
    # cv2.imshow('Imagee', new_S)
    # cv2.imshow('Image', gray_img)
    # cv2.waitKey(0)
    return new_S, spts2, d

def boardDetect(img):
  M, ideal_grid, grid_next, grid_good, spts, d, S= findChessboard(img)
  best_lines_x = []
  best_lines_y = []
  # View
  if M is not None:
      M, _ = generateNewBestFit((ideal_grid+8)*32, grid_next, grid_good)
      img_warp = cv2.warpPerspective(img, M, (17*32, 17*32), flags=cv2.WARP_INVERSE_MAP)
      best_lines_x, best_lines_y = getBestLines(img_warp)
      for line in best_lines_x:
          cv2.line(img_warp, (line, 0), (line, img_warp.shape[0]), (0,0,255),3)
      for line in best_lines_y:
          cv2.line(img_warp, (0, line), (img_warp.shape[1], line), (0,255,0),3)

      return M, best_lines_x, best_lines_y, img_warp, d, S
  else:
      return None, None, None, None, d, S

def main_func(FILE, test=False):

  img = loadImage(FILE)
  h, w = img.shape
  output, image_np = detection_img(FILE)
  M, best_lines_x, best_lines_y, img_warp, d, S  = boardDetect(img)
  points = []
  colors = []
  classes = []
  maps = ['p','r','n','b','q','k']
  board = chess.Board(fen=None)
  if M is None and test==False:
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(image_np,'Failed to detect grid',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
      return image_np
  if M is None and test==True:
      with open("fail.txt", "a") as text_file:
          text_file.write(FILE + '\n')
      path2 = 'test/2_detection/'+os.path.basename(FILE)
      imsave(path2, image_np)
      return None
  if len(output) > 0:
      for box, pClass in output:

          ymin, xmin, ymax, xmax = box

          xmin = int(xmin * w)
          xmax = int(xmax * w)
          ymin = int(ymin * h)
          ymax = int(ymax * h)

          temp_img = None
          temp_img = img[ymin: ymax, xmin:xmax]
          mean_intensity = np.mean(temp_img)
          c = maps[pClass-1]
          if (mean_intensity > 127):
              c = c.upper()
          # print(mean_intensity)
          y_point = ymax
          x_point = int((xmin + xmax)/2)
          point = [x_point, y_point]
          points.append(point)
          classes.append(c)

      points_np = np.array([points], dtype=np.float32)
      warped_points = cv2.perspectiveTransform(points_np, np.linalg.inv(M)).squeeze()
      if warped_points.ndim < 2:
          warped_points = warped_points[np.newaxis]

      letters = ['b', 'c', 'd', 'e', 'f','g','h']
      maps = ['p','r','n','b','q','k']
      print(d)
      locations = []

      for i, pt in enumerate(warped_points):

          l = 'a'
          for j in range(len(best_lines_x)):
              if pt[0] > best_lines_x[j]:
                  l = letters[j]
              else:
                  break
          n = 1
          for k in range(len(best_lines_y)):
              if pt[1] > int(best_lines_y[k] - d/8):
                  n = k + 2
              else:
                  break
          location = l + str(n)
          if location not in locations:
              locations.append(location)
          else:
              locations.append('XX')
          print (classes[i],location)


      for i in range(len(classes)):
          if locations[i] == 'XX':
              continue
          piece = chess.Piece.from_symbol(classes[i])
          board.set_piece_at(chess.SQUARE_NAMES.index(locations[i]),piece)
  bfen = board.board_fen()
  svg = chess.svg.board(board=board)
  with open("temp/OutputSVG.svg", "w") as text_file:
      text_file.write(svg)
  drawing = svg2rlg("temp/OutputSVG.svg")
  renderPM.drawToFile(drawing, "temp/OutputPNG.jpg")
  outputImg =  np.array(PIL.Image.open("temp/OutputPNG.jpg").convert('RGB') )
  if test is True:
      path1 = 'test/1_saddle/'+os.path.basename(FILE)
      imsave(path1, S)
      path2 = 'test/2_detection/'+os.path.basename(FILE)
      imsave(path2, image_np)
      path3 = 'test/3_fit_grid/' + os.path.basename(FILE)
      imsave(path3, img_warp)
      path4 = 'test/4_result/' + os.path.basename(FILE)
      imsave(path4, outputImg)
      return None
  else:

      return outputImg
####################################################################################################################################
### End: Written code: Saddle point detection and elimination from improved Hessian matrix, and integration with piece detection ###
####################################################################################################################################
