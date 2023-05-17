import cv2
import skimage as sk
import numpy as np

def sift(img_paths):
    extractor = cv2.xfeatures2d.SIFT_create()

    path_des = []
    path_key = []
    for path in img_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
        path_key.append((path,img_keypoints))
        path_des.append((path, img_descriptors))
    
    descriptor_sep = sep_descriptors(path_des)

    return path_des, descriptor_sep, path_key, extractor

def orb(img_paths):
    extractor = cv2.ORB_create()

    path_des = []
    path_key = []
    for path in img_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
        path_key.append((path,img_keypoints))
        path_des.append((path, img_descriptors))
    
    descriptor_sep = sep_descriptors(path_des)
        
    return path_des, descriptor_sep, path_key, extractor

def sep_descriptors(path_descriptors):
    descriptors = path_descriptors[0][1]

    for _,descriptor in path_descriptors[1:]:
        descriptors=np.vstack((descriptors,descriptor))

    descriptors = descriptors.astype(float)

    return descriptors

def hog(img_paths): #unused
    hog = cv2.HOGDescriptor()
    
    path_des = []
    for path in img_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_descriptors = hog.compute(img)
        path_des.append((path, img_descriptors))
        
    return path_des, hog

def reduce_hog_dimensions(descriptors): #unfinished, had trouble working with HoG
    pass



def sift_adl():
    # unfinished SIFT feature extractor I was attempting to make
    # I followed Lowe's paper and another paper dedicated to deconstructing SIFT alg,
    def create_dogs(img, s, sigma=1.6):
        img = np.array(img)
        #determine the number of needed octaves
        octaves = int(round(np.log(min([img.shape[0],img.shape[1]])) / np.log(2) - 1))
        #choose the number of intervals for each octave(can be any integer)
        num_imgs = s + 3 #from Lowe's paper
        #generate the sigmas for each interval
        k = 2**(1/s)
        sigmas = np.zeros(num_imgs)
        sigmas[0] = sigma
        for i in range(1, len(sigmas)):
            p = k**(i-1)*sigma
            sigmas[i] = np.sqrt((k*p)**2 - p**2)
    #     print(sigmas)
    #     print(octaves)
        guass_imgs = []
        dog_imgs = []
        new_img = img[:]
        for i in range(octaves):
            oct_guass_imgs = []
            oct_dog_imgs = []
            for j in range(num_imgs):                
                #apply gaussian using corresponding sigma
                blurred_img = cv2.GaussianBlur(new_img, [0,0], sigmaX=sigmas[j])
                oct_guass_imgs.append(blurred_img)
            guass_imgs.append(oct_guass_imgs)
            for j in range(num_imgs-1,1,-1):
                #gather dog imgs
                dog = np.subtract(oct_guass_imgs[j], oct_guass_imgs[j-1])
    #             print(dog)
                oct_dog_imgs.append(dog)
            dog_imgs.append(oct_dog_imgs)
            new_img = np.resize(new_img, (int(img.shape[0]/2), int(img.shape[1]/2)))
    #     print(dog_imgs)
        return dog_imgs, guass_imgs
            
    def find_cand_keypoints(dog_imgs):
        keypoints = []
        scales_per_octave = len(dog_imgs[0])
        thresh = (2**(1/scales_per_octave)-1)/(2**(1/3)-1)
        for i in range(len(dog_imgs)):
            oct_dog_imgs = dog_imgs[i]
            for layer in range(1,len(dog_imgs[0])-1):
                img_bel = oct_dog_imgs[layer-1]
                img_mid = oct_dog_imgs[layer]
                img_abov = oct_dog_imgs[layer+1]
    #             print(img_mid)
                for j in range(1, img_bel.shape[0]-1):
                    for k in range(1, img_bel.shape[1]-1):
                        sub_a = img_bel[j-1:j+2,k-1:k+2]
                        sub_b = img_mid[j-1:j+2,k-1:k+2]
                        sub_c = img_abov[j-1:j+2,k-1:k+2]
                        sub_b_mod = sub_b[:]
                        sub_b_mod[1,1] = sub_b_mod[1,0]
    #                     print(sub_a)
    #                     print(sub_b)
    #                     print(sub_c)

                        if (np.abs(sub_b[1,1]) < 0.8*thresh) and ((sub_b[1,1] > np.max(sub_a) and sub_b[1,1] > np.max(sub_c) and sub_b[1,1] > np.max(sub_b_mod)) or \
                        (sub_b[1,1] < np.min(sub_a) and sub_b[1,1] < np.min(sub_c) and sub_b[1,1] < np.min(sub_b_mod))):
    #                         print(True)
                            keypoints.append((i,layer,j,k))
        
        return keypoints
    # dog, guass = create_dogs(tst_img, 3, 1.6)
    # keypoints = find_cand_keypoints(dog)
    # print(keypoints)
    # t = detect_ss_extrema(tst_img, 3, 1.6)
    # t = t[0]
    # a = t[0]
    # b = t[1]
    # c = t[2]
    # i = 165
    # j = 254
    # sub_a = a[i-1:i+2,j-1:j+2]
    # sub_b = b[i-1:i+2,j-1:j+2]
    # sub_c = c[i-1:i+2,j-1:j+2]
    # print((sub_a))
    # print(np.max(sub_a),np.min(sub_a))
    # sub_b_mod = sub_b[:]
    # print(sub_b)
    # sub_b_mod[1,1] = sub_b_mod[1,0]
    # print(sub_b_mod)
    # print(np.max(sub_b_mod),np.min(sub_b_mod))
    # print(sub_c)
    # print(np.max(sub_c),np.min(sub_c))

    # if (sub_b[1,1] > np.max(sub_a) and sub_b[1,1] > np.max(sub_c) and sub_b[1,1] > np.max(sub_b_mod)) or \
    #     (sub_b[1,1] < np.min(sub_a) and sub_b[1,1] < np.min(sub_c) and sub_b[1,1] < np.min(sub_b_mod)):
    #     print(True)
    def quadratic_interp(img_abov, img_mid, img_bel, m, n):
    #     print(img_abov)
    #     print(img_abov[m])
    #     print(img_abov[m,n])
        oct_gradient = [(img_abov[m,n]-img_bel[m,n])/2, #s
                        (img_mid[m+1,n]-img_mid[m-1,n])/2, #x
                        (img_mid[m,n+1]-img_mid[m,n-1])/2] #y
        def d(point1,point2,point3): #computing diagonals in hessian
            return point1 + point2 - 2*point3
        
        def c(point1,point2,point3,point4): #computing corners in hessian
            return (point1 - point2 - point3 + point4)/4
        
        h11 = d(img_abov[m,n],img_bel[m,n], img_mid[m,n])
        h22 = d(img_mid[m+1,n],img_mid[m-1,n], img_mid[m,n])
        h33 = d(img_mid[m,n+1],img_mid[m,n-1], img_mid[m,n])
        h12 = c(img_abov[m+1,n],img_abov[m-1,n],img_bel[m+1,n],img_bel[m-1,n])
        h13 = c(img_abov[m,n+1],img_abov[m,n-1],img_bel[m,n+1],img_bel[m,n-1])
        h23 = c(img_mid[m+1,n+1],img_mid[m+1,n-1],img_mid[m-1,n+1],img_mid[m-1,n-1])
        
        hessian = np.array([[h11,h12,h13],[h12,h22,h23],[h13,h23,h33]]) #not used but just in case
        
        det = h22*h33*h11 - h22*h13*h13 - h23*h23*h11 + 2*h23*h12*h13 - h12*h12*h33
        aa = (h33*h11 - h13*h13)/det
        ab = (h12*h13 - h23*h11)/det
        ac = (h23*h13 - h12*h33)/det
        bb = (h22*h11 - h12*h12)/det
        bc = (h23*h12 - h22*h13)/det
        cc = (h22*h33 - h23*h23)/det
        #offsets
        a = np.zeros(3)
        a[1] = -aa*oct_gradient[1] - ab*oct_gradient[2] - ac*oct_gradient[0] #x
        a[2] = -ab*oct_gradient[1] - bb*oct_gradient[2] - bc*oct_gradient[0] #y
        a[0] = -ac*oct_gradient[1] - bc*oct_gradient[2] - cc*oct_gradient[0] #scale
        
        w = 0.5*(oct_gradient[1]*a[1] + oct_gradient[2]*a[2] + oct_gradient[0]*a[0])  
        
        return a, w, hessian, oct_gradient
        
    # a, w, h, g = quadratic_interp(dog[0][1+1].astype('float32')/255,dog[0][1].astype('float32')/255,dog[0][1-1].astype('float32')/255,1,299)
    # print(dog[0][1].shape)
    # print("a=",a)
    # print("w=",w)
    # print("h", h)
    # print('g',g)
    # e = -np.linalg.lstsq(h, g, rcond=None)[0]
    # print('e',e)
    # print(np.max(np.abs(a[0]),np.abs(a[1]),np.abs(a[2])))

    def keypoint_interp(keypoints, dog_imgs, sigma_min=1.6):
        new_keypoints = []
        for point in keypoints:
            s,m,n = point[1], point[2], point[3]
            img_height,img_width = dog_imgs[point[0]][0].shape
            
            num_interp = 0
            apnd = False
            while num_interp < 5:
                a, w, H, G = quadratic_interp(dog_imgs[point[0]][point[1]+1].astype('float32')/255, 
                                        dog_imgs[point[0]][point[1]].astype('float32')/255,
                                        dog_imgs[point[0]][point[1]-1].astype('float32')/255,
                                        int(m),int(n))
                
                sig = max(point[0]*2,1) * sigma_min * 2**(a[0]+s)
                x = max(point[0]*2,1) * (a[1]+m)
                y = max(point[0]*2,1) * (a[2]+n)
                
                if np.isnan(a[0]) or np.isnan(a[1]) or np.isnan(a[2]):
                    break
                s += int(round(a[0]))
                m += int(round(a[1]))
                n += int(round(a[2]))
                print(m,n)
                if max(abs(a[1]),abs(a[1]),abs(a[2])) < 0.6:
                    apnd = True
                    break 
                if n >= img_width or m >= img_height or m < 0 or n < 0 or m is not int or n is not int:
                    break
                num_interp += 1
        
            if apnd:
                new_keypoints.append((point[0],int(s),int(m),int(n), sig, int(x), int(y), w))
        
        return new_keypoints

    # new_keypoints = keypoint_interp(keypoints, dog)
    # print(new_keypoints)

    def orientation_assignment():
        pass

    def keypoint_descriptor():
        pass


