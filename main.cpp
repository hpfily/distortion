
#include <iostream>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define Camera_W 640  
#define Camera_H 480
using namespace cv;
using namespace std;
Mat Camera_Matrix,Distortion_Coefficients;


void crcTodis(double *crcU,double *crcV,double *disU,double *disV)
{
	static double k1=Distortion_Coefficients.at<double>(0,0);
	static double k2=Distortion_Coefficients.at<double>(1,0);
	static double k3=Distortion_Coefficients.at<double>(4,0);
	static double fx=Camera_Matrix.at<double>(0,0);
	static double fy=Camera_Matrix.at<double>(1,1);
	static double cx=Camera_Matrix.at<double>(0,2);
	static double cy=Camera_Matrix.at<double>(1,2);

	double u_corrected=*crcU;
	double v_corrected=*crcV;

	double x_corrected=(u_corrected-cx)/fx;
	double y_corrected=(v_corrected-cy)/fy;
	double r2=pow(x_corrected,2)+pow(y_corrected,2);
	double r4=pow(r2,2);
	double r6=r2*r4;
	double x_distortion=x_corrected*(1+k1*r2+k2*r4+k3*r6);
	double y_distortion=y_corrected*(1+k1*r2+k2*r4+k3*r6);
	double u_distortion=x_distortion*fx+cx;
	double v_distortion=y_distortion*fy+cy;

	*disU = u_distortion;
	*disV = v_distortion;
}

int my_distortion()
{
	
	double k1=Distortion_Coefficients.at<double>(0,0);
	double k2=Distortion_Coefficients.at<double>(1,0);
	double k3=Distortion_Coefficients.at<double>(4,0);
	cout<<"k1="<<k1<<"    k2="<<k2<<"      k3="<<k3<<endl;
	double fx=Camera_Matrix.at<double>(0,0);
	double fy=Camera_Matrix.at<double>(1,1);
	double cy=Camera_Matrix.at<double>(0,2);
	double cx=Camera_Matrix.at<double>(1,2);
	cout<<"fx="<<fx<<"    fy="<<fy<<"     cx="<<cx<<"      cy="<<cy<<endl;
	Mat src=imread("image/1.bmp",0);
	Mat dst(550,750,CV_8UC1,Scalar(50));
	
	imshow("my_distortion",src);
	for (int nx=0;nx<479;nx++)
	{

		for (int ny=0;ny<639;ny++)
		{
			
			double x=(nx-cx)/fx;
			double y=(ny-cy)/fy;

			double r2=pow(x,2)+pow(y,2);
			double r4=pow(r2,2);
			double r6=r2*r4;
			double xx=x*(1-k1*r2-k2*r4-k3*r6);
			double yy=y*(1-k1*r2-k2*r4-k3*r6);
//			double xx=x*(1-k1*r2);
//			double yy=y*(1-k1*r2);
			double u=xx*fx+cx;
			double v=yy*fy+cy;
			if (nx==450)
			{
				if (ny%20==0)
				{
					cout<<"nx="<<nx<<"   ny="<<ny<<"   r2="<<r2<<endl;
					cout<<"xx="<<xx<<"   yy="<<yy<<endl;
					cout<<"u="<<u<<"   v="<<v<<endl;
				}
			}
//			cout<<"xx="<<xx<<"   yy="<<yy<<endl;
//			cout<<"u="<<u<<"   v="<<v<<endl;
			if (u>0&&u<500&&v<680)
			{
				int temp=src.at<uchar>(nx,ny);
				dst.at<uchar>(u,v+60)=temp;
			}
			
		}
		

	}
	imshow("my_distortion3",dst);
	for (int nx=10;nx<490;nx++)
	{
		for (int ny=10;ny<690;ny++)
		{
			int temp=dst.at<uchar>(nx,ny);
			if (temp==50)
			{
				int dtemp=dst.at<uchar>(nx+3,ny);
				if (dtemp!=50)
				{
					dst.at<uchar>(nx,ny)=dst.at<uchar>(nx+3,ny);
				}
				else
				{
					dst.at<uchar>(nx,ny)=dst.at<uchar>(nx,ny+3);
				}
			}
		}
	}
	imshow("my_distortion5",dst);
	Mat ddst(480,640,CV_8UC1);
	IplImage *Isrc=&IplImage(dst);
	IplImage *Idst=&IplImage(ddst);
	cvResize(Isrc,Idst);
	imshow("my_distortion7",ddst);

	return 0;
}
int my_distortion_2()
{
	double k1=Distortion_Coefficients.at<double>(0,0);
	double k2=Distortion_Coefficients.at<double>(1,0);
	double k3=Distortion_Coefficients.at<double>(4,0);
	cout<<"k1="<<k1<<"    k2="<<k2<<"      k3="<<k3<<endl;
	double fx=Camera_Matrix.at<double>(0,0);
	double fy=Camera_Matrix.at<double>(1,1);
	double cx=Camera_Matrix.at<double>(0,2);
	double cy=Camera_Matrix.at<double>(1,2);
	cout<<"fx="<<fx<<"    fy="<<fy<<"     cx="<<cx<<"      cy="<<cy<<endl;
	Mat src=imread("image/8.bmp",0);
	Mat dst(480,640,CV_8UC1,Scalar(50));
	
	int  (*UV_map_p)[Camera_H][2] = new int[Camera_W][Camera_H][2];
	if (!UV_map_p)
	{
		cout<<"内存没有申请到鸟"<<endl;
	}
	memset(UV_map_p,0,sizeof(int)*Camera_H*Camera_W*2);


	imshow("my_distortion",src);
	

	for (double v_corrected=0;v_corrected<480;v_corrected++)
	{
		for (double u_corrected=0;u_corrected<640;u_corrected++)
		{
		/*
			double x_corrected=(u_corrected-cx)/fx;
			double y_corrected=(v_corrected-cy)/fy;
			double r2=pow(x_corrected,2)+pow(y_corrected,2);
			double r4=pow(r2,2);
			double r6=r2*r4;
			double x_distortion=x_corrected*(1+k1*r2+k2*r4+k3*r6);
			double y_distortion=y_corrected*(1+k1*r2+k2*r4+k3*r6);
			double u_distortion=x_distortion*fx+cx;
			double v_distortion=y_distortion*fy+cy;
		//	double u=nx+(nx-cx)*(k1*r2+k2*r4+k3*r6);
		//	double v=ny+(ny-cy)*(k1*r2+k2*r4+k3*r6);
		*/
			double u_distortion,v_distortion;
			crcTodis(&u_corrected,&v_corrected,&u_distortion,&v_distortion);

			if (u_distortion>0&&u_distortion<639&&v_distortion>0&&v_distortion<479)
			{
				u_distortion=(int)(u_distortion+0.5);
				v_distortion=(int)(v_distortion+0.5);
				int temp=src.at<uchar>(v_distortion,u_distortion);
				dst.at<uchar>(v_corrected,u_corrected)=temp;
				UV_map_p[(int)u_distortion][(int)v_distortion][0]=u_corrected;
				UV_map_p[(int)u_distortion][(int)v_distortion][1]=v_corrected;
				if (u_distortion==400&&v_distortion==400)
				{
					cout<<"u_corrected="<<u_corrected<<"   "<<"v_corrected="<<v_corrected<<endl;
				}
			}

		}
	}

	imshow("my_distortion2",dst);
	fstream fs;
	fs.open("data.dat",ios_base::out|ios_base::binary);
	if(!fs)
	{
		cout<<"打开文件失败"<<endl;
	}
	else
	{
		cout<<"打开文件成功"<<endl;
	}
	fs.write((char*)UV_map_p,sizeof(int)*Camera_H*Camera_W*2);
	fs.close();

	delete  []UV_map_p;
	UV_map_p=NULL;
	return 0;
}


int my_distortion_3()
{
	Mat src=imread("image/8.bmp",0);
	Mat dst(480,640,CV_8UC1,Scalar(50));

	for (double u_corrected=0;u_corrected<480;u_corrected++)
	{
		for (double v_corrected=0;v_corrected<640;v_corrected++)
		{
			double u_distortion,v_distortion;
			crcTodis(&u_corrected,&v_corrected,&u_distortion,&v_distortion);

			if (u_distortion>0&&u_distortion<479&&v_distortion>0&&v_distortion<639)
			{
//				u_distortion=(int)(u_distortion+0.5);
//				v_distortion=(int)(v_distortion+0.5);
				int temp=src.at<uchar>(u_distortion,v_distortion);
				dst.at<uchar>(u_corrected,v_corrected)=temp;
			}

		}
	}
	imshow("my_distortion_3",dst);
	
	return 0;
}
struct Min
{
	double uMin;
	double vMin;
	double lenMin;
};

void getMapdata()
{
	double disU,disV,crcU,crcV;

	int  (*UV_map_p)[Camera_H][2]=new int[Camera_W][Camera_H][2];
	if (!UV_map_p)
	{
		cout<<"内存没有申请到鸟"<<endl;
	}
	memset(UV_map_p,0,sizeof(int)*Camera_H*Camera_W*2);
	fstream fs("data.dat",ios_base::in|ios_base::out|ios_base::binary);
	fs.read((char*)UV_map_p,sizeof(int)*Camera_H*Camera_W*2);
	cout<<"gcount"<<fs.gcount()<<endl;

	for (disU=400;disU<=403;disU++)
	{
		for (disV=400;disV<=403;disV++)
		{

			crcU=UV_map_p[(int)disU][(int)disV][0],crcV=UV_map_p[(int)disU][(int)disV][1];

			cout<<"("<<disU<<","<<disV<<")"<<"      "<<"("<<crcU<<","<<crcV<<")"<<endl;
		}
	}
	cout<<"================================"<<endl;
	disU=500,disV=400;
	crcU=UV_map_p[(int)disU][(int)disV][0],crcV=UV_map_p[(int)disU][(int)disV][1];

	cout<<"("<<crcU<<","<<crcV<<")"<<"      "<<"("<<disU<<","<<disV<<")"<<endl;
	cout<<"================================"<<endl;

	for (int i=0;i<3;crcU++,i++)
	{
		for (int j=0;j<3;crcV++,j++)
		{

			crcTodis(&crcU,&crcV,&disU,&disV);

			cout<<"("<<crcU<<","<<crcV<<")"<<"      "<<"("<<disU<<","<<disV<<")"<<endl;
		}
	}
	cout<<"================================"<<endl;

	disU=50,disV=50;
	crcU=UV_map_p[(int)disU][(int)disV][0],crcV=UV_map_p[(int)disU][(int)disV][1];
	cout<<"("<<crcU<<","<<crcV<<")"<<"      "<<"("<<disU<<","<<disV<<")"<<endl;
	double crcUtmp = crcU-1;
	double crcVtmp = crcV-1;
	struct Min m;
	m.lenMin=100;
	while(crcUtmp<=crcU+1)
	{
		while(crcVtmp<=crcV+1)
		{
			double disUtmp,disVtmp;
			crcTodis(&crcUtmp,&crcVtmp,&disUtmp,&disVtmp);
			double len=(disUtmp-disU)*(disUtmp-disU)+(disVtmp-disV)*(disVtmp-disV);
			if (len<m.lenMin)
			{
				m.lenMin=len;
				m.uMin=crcUtmp;
				m.vMin=crcVtmp;
				cout<<"("<<crcUtmp<<","<<crcVtmp<<")"<<"      "<<"("<<disUtmp<<","<<disVtmp<<")"<<endl;
			}
			crcVtmp += 0.1f;
		}	
		crcUtmp += 0.1f;
		crcVtmp = crcVtmp-2;
	}
	delete []UV_map_p;
	UV_map_p = NULL;
	fs.close();
}


int main(int argc,char* argv[])
{
	
	FileStorage fs;
	fs.open("out_camera_data.xml",FileStorage::READ);
	
	fs["Camera_Matrix"]>>Camera_Matrix;
	fs["Distortion_Coefficients"]>>Distortion_Coefficients;

	my_distortion_2();
//	my_distortion_3();
	getMapdata();


	Mat ori=imread("image/8.bmp",CV_LOAD_IMAGE_GRAYSCALE);
	Size BoardSize;
	BoardSize.width=9;
	BoardSize.height=6;
	vector<Point2f> pointBuf;
	bool found=findChessboardCorners(ori,BoardSize,pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
//	cout<< Mat(pointBuf)<<endl;
// 	cout<<pointBuf.size()<<endl;
	if (found)
	{
		drawChessboardCorners(ori,BoardSize,Mat(pointBuf),found);
	}
	imshow("ori image",ori);
	Size imageSize=ori.size();
	
	Mat view, rview, map1, map2;
	
	initUndistortRectifyMap(Camera_Matrix, Distortion_Coefficients, Mat(),
		getOptimalNewCameraMatrix(Camera_Matrix, Distortion_Coefficients, imageSize, 1, imageSize, 0),
		imageSize, CV_16SC2, map1, map2);
	view=ori;
	remap(view, rview, map1, map2, INTER_LINEAR);
	imshow("Image View", rview);

	waitKey();
}
/*
using namespace cv;
using namespace std;
int g_nContrastValue;
int g_nBrightValue;
Mat g_srcImage, g_dstImage;
void callback(int ContrastValue,void*)
{
	printf("haha  g_nConstrastValue=%d \n",g_nContrastValue);
	printf("haha  g_nBrightValue=%d \n",g_nBrightValue);
	for (int y=0;y<g_srcImage.rows;++y)
	{
		for (int x=0;x<g_srcImage.cols;++x)
		{
			for (int c=0;c<3;++c)
			{
				g_dstImage.at<Vec3b>(y,x)[c]=saturate_cast<uchar>((ContrastValue*0.01)*g_srcImage.at<Vec3b>(y,x)[c]+g_nBrightValue);
			}
		}
	}
	imshow("TrackbarImage",g_dstImage);
}
int main(int argc,char* argv[])
{
	system("color5");
	g_srcImage=imread("pic.jpg");
	g_dstImage=Mat::zeros(g_srcImage.size(),g_srcImage.type());
	if (!g_srcImage.data)
	{
		printf("image read error!\n");
		waitKey();
		return -1;
	}
	imshow("image",g_srcImage);
	g_nContrastValue=50;
	g_nBrightValue=50;
	namedWindow("TrackbarImage");
	createTrackbar("Trackbar","TrackbarImage",&g_nContrastValue,300,callback);
	createTrackbar("TrackBright","TrackbarImage",&g_nBrightValue,200,callback);

	waitKey();

}
/*

int main(int argc, char* argv[])
{
	Mat src=imread("pic.jpg");
	imshow("image",src);
	vector<Mat> channels;
	split(src,channels);
	imshow("image2",channels.at(1));
	waitKey();
}
/*
int main(int argc, char* argv[])
{
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cerr<<"Can not open camera"<<endl;
		return -1;
	}
	Mat cam;
	Mat edge;
	while(1)
	{
		Mat frame;
		cap >> frame;
		if(frame.empty())
			break;
		cvtColor(frame,cam,CV_BGR2GRAY);
		Canny(cam,edge,0,30,3);
		imshow("org",frame);
		imshow("gray",cam);
		imshow("edge",edge);
		cvWaitKey(33);
		
	}
	waitKey();
	return 0;
}
/*
int main(int argc,char* argv[])
{
	Mat srcImag=imread("pic.jpg");
	Mat logoImag=imread("2.jpg");
	if (!srcImag.data||!logoImag.data)
	{
		fprintf(stdout,"文件读取错误\n");
	}
	Mat imageROI=srcImag(Rect(400,600,logoImag.cols,logoImag.rows));

//	addWeighted(imageROI,0.4,logoImag,0.6,0.0,imageROI);
	imshow("image",srcImag);

	Mat mask=imread("2.jpg",0);
	imshow("image1",mask);
	logoImag.copyTo(imageROI,mask);
	imshow("image",srcImag);

	waitKey();
}
/*
void createAlphaMat(Mat &mat)  
{  
	for(int i = 0; i < mat.rows; ++i) {  
		for(int j = 0; j < mat.cols; ++j) {  
			Vec4b& rgba = mat.at<Vec4b>(i, j);  
			rgba[0]= UCHAR_MAX;  
			rgba[1]= saturate_cast<uchar>((float (mat.cols - j)) / ((float)mat.cols) *UCHAR_MAX);  
			rgba[2]= saturate_cast<uchar>((float (mat.rows - i)) / ((float)mat.rows) *UCHAR_MAX);  
			rgba[3]= saturate_cast<uchar>(0.5 * (rgba[1] + rgba[2]));  
		}  
	}  
}  

int main(int argc,char* argv[])
{
	Mat mat(480,640,CV_8UC4);
	createAlphaMat(mat);
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	imwrite("透明Alpha值图.png",mat,compression_params);
	fprintf(stdout,"PNG图片保存完毕~\n");
	return 0;
	
}
	/*
	Mat img=imread("pic.jpg",0);
	imshow("image",img);
	waitKey();
	return 0;
}
	*/