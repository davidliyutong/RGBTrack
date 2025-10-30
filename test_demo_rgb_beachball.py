# ==========================================================
# Real-time Geometric & Color-based Monocular Beachball Pose Estimation
# (Based on your original FoundationPose pipeline structure)
# ==========================================================
import os, time, argparse, logging, cv2, imageio, numpy as np, trimesh
from tqdm import tqdm
# ==========================================================
# ----------- 几何 + 颜色沙滩球位姿估计函数 --------------
# ==========================================================
def hsv_mask(img_bgr, low, high):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, low, high)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    return m

def mask_color(bgr, spec):
    if isinstance(spec, list):
        m = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for (lo,hi) in spec:
            m = cv2.bitwise_or(m, hsv_mask(bgr, lo, hi))
        return m
    else:
        return hsv_mask(bgr, spec[0], spec[1])

def largest_circle(mask):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    (u,v),r = cv2.minEnclosingCircle(c)
    return (float(u),float(v),float(r))

def centroid(mask):
    M = cv2.moments(mask, binaryImage=True)
    if M["m00"]<1e-3: return None
    return (M["m10"]/M["m00"], M["m01"]/M["m00"])

def ray_from_px(u,v,fx,fy,cx,cy):
    x=(u-cx)/fx; y=(v-cy)/fy
    d=np.array([x,y,1.0],dtype=np.float32)
    return d/np.linalg.norm(d)

def intersect_sphere(C,R,d):
    oc=-C
    a=np.dot(d,d)
    b=2*np.dot(d,oc)
    c=np.dot(oc,oc)-R*R
    disc=b*b-4*a*c
    if disc<=1e-9: return None
    s1=(-b-np.sqrt(disc))/(2*a)
    s2=(-b+np.sqrt(disc))/(2*a)
    lam=s1 if s1>0 else s2 if s2>0 else None
    if lam is None: return None
    return lam*d

def orthonormal_basis(z_hat,x_ref):
    z=z_hat/np.linalg.norm(z_hat)
    x=x_ref - np.dot(x_ref,z)*z
    if np.linalg.norm(x)<1e-6:
        tmp=np.array([1,0,0],dtype=np.float32)
        if abs(np.dot(tmp,z))>0.9: tmp=np.array([0,1,0],dtype=np.float32)
        x=tmp - np.dot(tmp,z)*z
    x/=np.linalg.norm(x)
    y=np.cross(z,x)
    R=np.column_stack([x,y,z])
    if np.linalg.det(R)<0: R[:,1]=-R[:,1]
    return R.astype(np.float32)

def estimate_ball_pose(rgb_bgr,K,RADIUS,HSV,prev_pose=None,return_vis=True):
    fx,fy=float(K[0,0]),float(K[1,1]); cx,cy=float(K[0,2]),float(K[1,2])
    img=rgb_bgr
    # 合并掩码找圆
    m_all=np.zeros(img.shape[:2],dtype=np.uint8)
    for k in HSV['merge']:
        m_all=cv2.bitwise_or(m_all, mask_color(img,k))
    circ=largest_circle(m_all)
    if circ is None: return prev_pose if prev_pose is not None else np.eye(4,dtype=np.float32),img
    u,v,rpx=circ; rpx=max(rpx,1e-3)
    Z=fx*RADIUS/rpx
    X=(u-cx)*Z/fx; Y=(v-cy)*Z/fy
    C=np.array([X,Y,Z],dtype=np.float32)
    # 姿态
    R_cw=np.eye(3,dtype=np.float32)
    red_in=mask_color(img,HSV['red'])
    blue_in=mask_color(img,HSV['blue'])
    h,w=img.shape[:2]; yy,xx=np.ogrid[:h,:w]
    disk=((xx-u)**2+(yy-v)**2<=(rpx*rpx)).astype(np.uint8)*255
    red_in=cv2.bitwise_and(red_in,red_in,mask=disk)
    blue_in=cv2.bitwise_and(blue_in,blue_in,mask=disk)
    cr=centroid(red_in); cb=centroid(blue_in)
    if cr and cb:
        dr=ray_from_px(*cr,fx,fy,cx,cy)
        db=ray_from_px(*cb,fx,fy,cx,cy)
        Pr=intersect_sphere(C,RADIUS,dr)
        Pb=intersect_sphere(C,RADIUS,db)
        if Pr is not None and Pb is not None:
            pr=(Pr-C)/RADIUS; pb=(Pb-C)/RADIUS
            R_cw=orthonormal_basis(pr,pb)
    pose=np.eye(4,dtype=np.float32); pose[:3,:3]=R_cw; pose[:3,3]=C
    if not return_vis: return pose,None
    vis=img.copy()
    cv2.circle(vis,(int(u),int(v)),int(rpx),(0,255,0),2)
    cv2.putText(vis,f"Z={Z:.3f}m",(int(u)-40,int(v)-int(rpx)-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    for a,color in [(R_cw[:,0],(255,0,0)),(R_cw[:,1],(0,255,255)),(R_cw[:,2],(0,0,255))]:
        tip=C+a*RADIUS*1.2
        u2=fx*tip[0]/tip[2]+cx; v2=fy*tip[1]/tip[2]+cy
        cv2.line(vis,(int(u),int(v)),(int(u2),int(v2)),color,2)
    return pose,vis

# ==========================================================
# ----------- 主程序入口 -----------------------------------
# ==========================================================
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    code_dir=os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--test_scene_dir",type=str,default=f"{code_dir}/demo_data/test")
    parser.add_argument("--save_video",type=int,default=1)
    args=parser.parse_args()
    SAVE_VIDEO=bool(args.save_video)
    # 输入路径与内参
    color_dir=os.path.join(args.test_scene_dir,"rgb")
    K=np.loadtxt(os.path.join(args.test_scene_dir,"cam_K.txt"))
    color_files=sorted([os.path.join(color_dir,f) for f in os.listdir(color_dir) if f.endswith((".png",".jpg"))])
    output_dir=os.path.join(code_dir,"output"); os.makedirs(output_dir,exist_ok=True)
    video_path=os.path.join(output_dir,"pose_overlay.mp4")
    poses_txt=os.path.join(output_dir,"poses.txt")
    video_writer=None
    # 球参数与HSV
    BALL_RADIUS=0.15
    HSV={
        'red':[(np.array([0,90,80]),np.array([10,255,255])),(np.array([170,90,80]),np.array([179,255,255]))],
        'blue':(np.array([100,90,80]),np.array([130,255,255])),
        'yellow':(np.array([20,90,80]),np.array([35,255,255])),
        'white':(np.array([0,0,200]),np.array([179,60,255]))
    }
    HSV['merge']=[HSV['red'],HSV['blue'],HSV['yellow'],HSV['white']]
    prev_pose=None
    print(f"Processing {len(color_files)} frames ...")
    with open(poses_txt,"w") as fpose:
        for idx,fp in enumerate(tqdm(color_files)):
            color=cv2.imread(fp)
            t1=time.time()
            pose,vis=estimate_ball_pose(color,K,BALL_RADIUS,HSV,prev_pose)
            t2=time.time(); prev_pose=pose
            np.savetxt(fpose,pose.reshape(4,4),fmt="%.6f"); fpose.write("\n")
            fps=int(1.0/max(t2-t1,1e-6))
            cv2.putText(vis,f"fps {fps}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            if SAVE_VIDEO:
                if video_writer is None:
                    h,w=vis.shape[:2]; fourcc=cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer=cv2.VideoWriter(video_path,fourcc,30,(w,h))
                video_writer.write(vis)
    if video_writer: video_writer.release()
    print(f"Saved video: {video_path}")
    print(f"Saved poses: {poses_txt}")
