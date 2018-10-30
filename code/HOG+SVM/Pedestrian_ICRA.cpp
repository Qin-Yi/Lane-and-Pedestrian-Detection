// by Jianxin Wu 2011 (wujx2001@ntu.edu.sg / wujx2001@gmail.com)

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <vector>
#include <climits>
#if defined ( _WIN32 )
    #include <io.h>
#else
    #include <unistd.h>
#endif
#include <sys/types.h>
#include <sys/timeb.h>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/ml/ml.hpp>  

#include "Pedestrian.h"

const int HUMAN_height = 108;
const int HUMAN_width = 36;
const int HUMAN_xdiv = 9;
const int HUMAN_ydiv = 4;
static const int EXT = 1;

std::vector <cv::Rect> found_mHOG;
std::vector <cv::Rect> found_dHOG;
std::vector <CRect> found_mC4;
std::vector <CRect> found_save;

// The detector
DetectionScanner scanner(HUMAN_height,HUMAN_width,HUMAN_xdiv,HUMAN_ydiv,256,0.8);
cv::HOGDescriptor mHOG(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
cv::HOGDescriptor dHOG(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);

// ---------------------------------------------------------------------
// Helper functions

// for timing of the detector

static timeb TimingMilliSeconds;

void StartOfDuration()
{
    ftime(&TimingMilliSeconds);
}

int EndOfDuration()
{
    struct timeb now;
    ftime(&now);
    return int( (now.time-TimingMilliSeconds.time)*1000+(now.millitm-TimingMilliSeconds.millitm) );
}

// compute the Sobel image "ct" from "original"
void ComputeCT(IntImage<double>& original,IntImage<int>& ct)
{
	ct.Create(original.nrow,original.ncol);
	for(int i=2;i<original.nrow-2;i++)
	{
		double* p1 = original.p[i-1];
		double* p2 = original.p[i];
		double* p3 = original.p[i+1];
		int* ctp = ct.p[i];
		for(int j=2;j<original.ncol-2;j++)
		{
			int index = 0;
			if(p2[j]<=p1[j-1]) index += 0x80;
			if(p2[j]<=p1[j]) index += 0x40;
			if(p2[j]<=p1[j+1]) index += 0x20;
			if(p2[j]<=p2[j-1]) index += 0x10;
			if(p2[j]<=p2[j+1]) index += 0x08;
			if(p2[j]<=p3[j-1]) index += 0x04;
			if(p2[j]<=p3[j]) index += 0x02;
			if(p2[j]<=p3[j+1]) index ++;
			ctp[j] = index;
		}
	}
}

// Load SVM models -- linear SVM trained using LIBLINEAR
double UseSVM_CD_FastEvaluationStructure(const char* modelfile, const int m, Array2dC<double>& result)
{
	std::ifstream in(modelfile);
	if(in.good()==false)
	{
		std::cout<<"SVM model "<<modelfile<<" can not be loaded."<<std::endl;
        exit(-1);
	}
	std::string buffer;
	std::getline(in,buffer); // first line
	std::getline(in,buffer); // second line
	std::getline(in,buffer); // third line
	in>>buffer; assert(buffer=="nr_feature");
	int num_dim;
	in>>num_dim; assert(num_dim>0 && num_dim==m);
	std::getline(in,buffer); // end of line 4
	in>>buffer; assert(buffer=="bias");
	int bias;
	in>>bias;
	std::getline(in,buffer); //end of line 5;
	in>>buffer; assert(buffer=="w");
	std::getline(in,buffer); //end of line 6
	result.Create(1,num_dim);
	for(int i=0;i<num_dim;i++) in>>result.buf[i];
	double rho = 0;
	if(bias>=0) in>>rho;
	in.close();
	return rho;
}

// Load SVM models -- Histogram Intersectin Kernel SVM trained by libHIK
double UseSVM_CD_FastEvaluationStructure(const char* modelfile, const int m, const int upper_bound, Array2dC<double>& result)
{
	std::ifstream in(modelfile);
	if(in.good()==false)
	{
		std::cout << "SVM model " << modelfile << " can not be loaded." << std::endl;
        exit(-1);
	}
	std::string buffer;
	std::getline(in,buffer); // first line
	std::getline(in,buffer); // second line
	std::getline(in,buffer); // third line
	in >> buffer; assert(buffer == "nr_feature");
	int num_dim;
	in >> num_dim; assert(num_dim > 0 && num_dim == m);
	std::getline(in,buffer); // end of line 4
	in >> buffer; assert(buffer == "bias");
	int bias;
	in>>bias;
	std::getline(in,buffer); //end of line 5;
	in >> buffer; assert(buffer == "w");
	std::getline(in,buffer); //end of line 6
	result.Create(num_dim, upper_bound);
	for (int i = 0; i < num_dim; i++)
		for (int j = 0; j < upper_bound; j++)
			in >> result.p[i][j];
	double rho = 0;
	if(bias>0)
	{
		in>>rho;
		in>>rho;
	}
	in.close();
	std::cout<<"Rho = "<<rho<<std::endl;
	return rho;
}

// find the intersection of "this" and "rect2", and put into "result"
bool CRect::Intersect(CRect& result,const CRect& rect2) const
{
    if( Empty() || rect2.Empty() ||
        left >= rect2.right || rect2.left >= right ||
        top >= rect2.bottom || rect2.top >= bottom )
    {
        result.Clear();
        return false;
    }
    result.left   = std::max( left, rect2.left );
    result.right  = std::min( right, rect2.right );
    result.top    = std::max( top, rect2.top );
    result.bottom = std::min( bottom, rect2.bottom );
    return true;
}

// find the union of "this" and "rect2", and put into "result"
bool CRect::Union(CRect& result,const CRect& rect2) const
{
    if(Empty())
    {
        if(rect2.Empty())
        {
            result.Clear();
            return false;
        }
        else
            result = rect2;
    }
    else
    {
        if(rect2.Empty())
            result = *this;
        else
        {
            result.left   = std::min( left, rect2.left );
            result.right  = std::max( right, rect2.right );
            result.top    = std::min( top, rect2.top );
            result.bottom = std::max( bottom, rect2.bottom );
        }
    }
    return true;
}

// A simple post-process (NMS, non-maximal suppression)
// "result" -- rectangles before merging
//          -- after this function it contains rectangles after NMS
// "combine_min" -- threshold of how many detection are needed to survive
void PostProcess(std::vector<CRect>& result,const int combine_min)
{
    std::vector<CRect> res1;
    std::vector<CRect> resmax;
    std::vector<int> res2;
    bool yet;
    CRect rectInter;

	for (unsigned int i = 0, size_i = result.size(); i < size_i; i++)
    {
        yet = false;
        CRect& result_i = result[i];
		for (unsigned int j = 0, size_r = res1.size(); j < size_r; j++)
        {
            CRect& resmax_j = resmax[j];
			if (result_i.Intersect(rectInter, resmax_j))
            {
				if (rectInter.Size() > 0.6*result_i.Size()
					&& rectInter.Size() > 0.6*resmax_j.Size()
					)
                {
                    CRect& res1_j = res1[j];
                    resmax_j.Union(resmax_j,result_i);
                    res1_j.bottom += result_i.bottom;
                    res1_j.top += result_i.top;
                    res1_j.left += result_i.left;
                    res1_j.right += result_i.right;
                    res2[j]++;
                    yet = true;
                    break;
                }
            }
        }
		if (yet == false)
        {
            res1.push_back(result_i);
            resmax.push_back(result_i);
            res2.push_back(1);
        }
    }

	for (unsigned int i = 0, size = res1.size(); i < size; i++)
    {
        const int count = res2[i];
        CRect& res1_i = res1[i];
        res1_i.top /= count;
        res1_i.bottom /= count;
        res1_i.left /= count;
        res1_i.right /= count;
    }

    result.clear();
	for (unsigned int i = 0, size = res1.size(); i < size; i++)
		if (res2[i] > combine_min)
            result.push_back(res1[i]);
}

// If one detection (after NMS) is inside another, remove the inside one
void RemoveCoveredRectangles(std::vector<CRect>& result, double threshold)
{
	std::vector<bool> covered;
	covered.resize(result.size());
	std::fill(covered.begin(), covered.end(), false);
	CRect inter;
	for (unsigned int i = 0; i < result.size(); i++)
	{
		for (unsigned int j = i + 1; j < result.size(); j++)
		{
			result[i].Intersect(inter, result[j]);
			double isize = sqrt(inter.Size());
			double weight1 = isize / sqrt(result[i].Size());
			double weight2 = isize / sqrt(result[j].Size());
			if (MAX(weight1, weight2) > threshold)
			{
				weight1 += result[i].weight * 1.5;
				weight2 += result[j].weight * 1.5;
				if (weight1 > weight2)
				{
					covered[j] = true;
				}
				else
				{
					covered[i] = true;
				}
			}
		}
	}
	std::vector<CRect> newresult;
	for (unsigned int i = 0; i < result.size(); i++)
		if (!covered[i])
			newresult.push_back(result[i]);
	result.clear();
	result.insert(result.begin(), newresult.begin(), newresult.end());
	newresult.clear();
}

// End of Helper functions
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Functions that load the two classifiers
void LoadCascade(DetectionScanner& ds)
{
	std::vector<NodeDetector::NodeType> types;
	std::vector<int> upper_bounds;
	std::vector<std::string> filenames;

	types.push_back(NodeDetector::CD_LIN); // first node
	upper_bounds.push_back(100);
	filenames.push_back("combined.txt.model");
	types.push_back(NodeDetector::CD_HIK); // second node
	upper_bounds.push_back(353);
	filenames.push_back("combined2.txt.model");
	
	ds.LoadDetector(types, upper_bounds, filenames);
	// You can adjust these parameters for different speed, accuracy etc
	ds.cascade->nodes[0]->thresh += 0.8;
	ds.cascade->nodes[1]->thresh -= 0.095;
}

void DetectionScanner::LoadDetector(std::vector<NodeDetector::NodeType>& types, std::vector<int>& upper_bounds, std::vector<std::string>& filenames)
{
	unsigned int depth = types.size();
	assert(depth > 0 && depth == upper_bounds.size() && depth == filenames.size());
	if(cascade)
		delete cascade;
	cascade = new CascadeDetector;
	assert(xdiv > 0 && ydiv > 0);
	for (unsigned int i = 0; i < depth; i++)
		cascade->AddNode(types[i], (xdiv - EXT)*(ydiv - EXT)*baseflength, upper_bounds[i], filenames[i].c_str());
		
	hist.Create(1, baseflength*(xdiv - EXT)*(ydiv - EXT));
}

void NodeDetector::Load(const NodeType _type, const int _featurelength, const int _upper_bound, const int _index, const char* _filename)
{
	type = _type;
	index = _index;
	filename = _filename;
	featurelength = _featurelength;
	upper_bound = _upper_bound;
	if (type == CD_LIN)
		thresh = UseSVM_CD_FastEvaluationStructure(_filename, _featurelength, classifier);
	else if (type == CD_HIK)
		thresh = UseSVM_CD_FastEvaluationStructure(_filename, _featurelength, upper_bound, classifier);
		
	if(type==CD_LIN) type = LINEAR;
	if(type==CD_HIK) type = HISTOGRAM;
}

void CascadeDetector::AddNode(const NodeDetector::NodeType _type, const int _featurelength, const int _upper_bound, const char* _filename)
{
	if(length==size)
	{
		int newsize = size * 2;
		NodeDetector** p = new NodeDetector*[newsize];
		assert(p != NULL);
		std::copy(nodes, nodes + size, p);
		size = newsize;
		delete[] nodes;
		nodes = p;
	}
	nodes[length] = new NodeDetector(_type, _featurelength, _upper_bound, length, _filename);
	length++;
}

// End of functions that load the two classifiers
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Detection functions

// initialization -- compute the Census Tranform image for CENTRIST
void DetectionScanner::InitImage(IntImage<double>& original)
{
	image = original;
	image.Sobel(sobel, false, false);
	ComputeCT(sobel, ct);
}

// combine the (xdiv-1)*(ydiv-1) integral images into a single one
void DetectionScanner::InitIntegralImages(const int stepsize)
{
	if (cascade->nodes[0]->type != NodeDetector::LINEAR)
		return; // No need to prepare integral images

	const int hd = height / xdiv * 2 - 2;
	const int wd = width / ydiv * 2 - 2;
	scores.Create(ct.nrow, ct.ncol);
	scores.Zero(cascade->nodes[0]->thresh / hd / wd);
	double* linearweights = cascade->nodes[0]->classifier.buf;
	for (int i = 0; i < xdiv - EXT; i++)
	{
		const int xoffset = height / xdiv*i;
		for (int j = 0; j < ydiv - EXT; j++)
		{
			const int yoffset = width / ydiv*j;
			for (int x = 2; x < ct.nrow - 2 - xoffset; x++)
			{
				int* ctp = ct.p[x + xoffset] + yoffset;
				double* tempp = scores.p[x];
				for (int y = 2; y < ct.ncol - 2 - yoffset; y++)
					tempp[y] += linearweights[ctp[y]];
			}
			linearweights += baseflength;
		}
	}
	scores.CalcIntegralImageInPlace();
	for (int i = 2; i < ct.nrow - 2 - height; i += stepsize)
	{
		double* p1 = scores.p[i];
		double* p2 = scores.p[i + hd];
		for (int j = 2; j < ct.ncol - 2 - width; j += stepsize)
			p1[j] += (p2[j + wd] - p2[j] - p1[j + wd]);
	}
}

// Resize the input image and then re-compute Sobel image etc
void DetectionScanner::ResizeImage()
{
	image.Resize(sobel,ratio); image.Swap(sobel);
	image.Sobel(sobel,false,false);
	ComputeCT(sobel,ct);
}

// The function that does the real detection
int DetectionScanner::FastScan(IntImage<double>& original, std::vector<CRect>& results, const int stepsize)
{
	if (original.nrow < height + 5 || original.ncol < width + 5) return 0;
	const int hd = height / xdiv;
	const int wd = width / ydiv;
	InitImage(original);
	results.clear();
	
	hist.Create(1, baseflength*(xdiv - EXT)*(ydiv - EXT));
	
	NodeDetector* node = cascade->nodes[1];
	double** pc = node->classifier.p;
	int oheight = original.nrow, owidth = original.ncol;
	CRect rect;
	while (image.nrow >= height && image.ncol >= width)
    {
		InitIntegralImages(stepsize);
		for (int i = 2; i + height < image.nrow - 2; i += stepsize)
        {
			const double* sp = scores.p[i];
			for (int j = 2; j + width < image.ncol - 2; j += stepsize)
            {
				if (sp[j] <= 0) continue;
				int* p = hist.buf;
				hist.Zero();
				for (int k = 0; k < xdiv - EXT; k++)
				{
					for (int t = 0; t < ydiv - EXT; t++)
					{
						for (int x = i + k*hd + 1; x < i + (k + 1 + EXT)*hd - 1; x++)
						{
							int* ctp = ct.p[x];
							for (int y = j + t*wd + 1; y < j + (t + 1 + EXT)*wd - 1; y++)
								p[ctp[y]]++;
						}
						p += baseflength;
					}
				}
				double score = node->thresh;
				for (int k = 0; k < node->classifier.nrow; k++) score += pc[k][hist.buf[k]];
				if (score > 0)
				{
					rect.top = i*oheight / image.nrow;
					rect.bottom = (i + height)*oheight / image.nrow;
					rect.left = j*owidth / image.ncol;
					rect.right = (j + width)*owidth / image.ncol;
					results.push_back(rect);
				}
            }
        }
		ResizeImage();
    }
	return 0;
}

// The interface function that detects pedestrians
// "filename" -- an input image
// detect pedestrians in image "filename" and save results to "outname"
// "ds" -- the detector cascade -- pass "scanner" as this parameter
// "out" -- file stream that saves the output rectangle information
int totaltime = 0;

std::vector <CRect> DetectHuman(IplImage* img, DetectionScanner& ds)
{
    std::vector<CRect> results;
    IntImage<double> original;
    original.Load(img);

	ds.FastScan(original,results,2);
    PostProcess(results,2);
    PostProcess(results,0);
	RemoveCoveredRectangles(results, 0.9);

	return results;
}

// End of Detection functions
// ---------------------------------------------------------------------
bool isSame(cv::Mat pic1, cv::Mat pic2)
{
	cv::resize(pic1, pic1, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);
	cv::resize(pic2, pic2, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);
	cv::cvtColor(pic1, pic1, CV_BGR2GRAY);
	cv::cvtColor(pic2, pic2, CV_BGR2GRAY);
	int iAvg1 = 0, iAvg2 = 0;
	int arr1[64], arr2[64];

	for (int i = 0; i < 8; i++)
	{
		uchar* data1 = pic1.ptr<uchar>(i);
		uchar* data2 = pic2.ptr<uchar>(i);

		int tmp = i * 8;

		for (int j = 0; j < 8; j++)
		{
			int tmp1 = tmp + j;

			arr1[tmp1] = data1[j] / 4 * 4;
			arr2[tmp1] = data2[j] / 4 * 4;

			iAvg1 += arr1[tmp1];
			iAvg2 += arr2[tmp1];
		}

		iAvg1 /= 64;
		iAvg2 /= 64;

		for (int i = 0; i < 64; i++)
		{
			arr1[i] = (arr1[i] >= iAvg1) ? 1 : 0;
			arr2[i] = (arr2[i] >= iAvg2) ? 1 : 0;
		}

		int iDiffNum = 0;

		for (int i = 0; i < 64; i++)
			if (arr1[i] != arr2[i])
				++iDiffNum;

		return iDiffNum <= 10;
	}
}
bool isLogical(CRect r, cv::Mat frame)
{
	bool result;
	CRect m;
	for (int i = 0; i < found_save.size(); i++)
	{
		found_save[i].Intersect(m, r);
		double isize = m.Size();
		if (isize > (r.Size() + found_save[i].Size()) * 0.2)
		{
			return true;
		}
	}
	for (int i = 0; i < found_save.size(); i++)
	{
		m = found_save[i];
		cv::Mat pic1 = frame(cv::Rect(m.left, m.top, m.right - m.left, m.bottom - m.top));
		cv::Mat pic2 = frame(cv::Rect(r.left, r.top, r.right - r.left, r.bottom - r.top));
		if (isSame(pic1, pic2))
		{
			return true;
		}
	}
	return false;
}
// Example of how to use the detector to detect from multiple input images
void RunFiles()
{
    LoadCascade(scanner);

	std::ifstream fin_detector("HOGDetectorForOpenCV.txt");
	float temp;
	std::vector<float> myDetector;//3781维的检测器参数  
	while (!fin_detector.eof())
	{
		fin_detector >> temp;
		myDetector.push_back(temp);//放入检测器数组  
	}
	mHOG.setSVMDetector(myDetector);
	dHOG.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

	std::cout<<"Detectors loaded."<<std::endl;
    
	cv::VideoCapture capture("live_left.avi");
	if (!capture.isOpened())
	{
		std::cout << "No Input Image";
		return;
	}

	bool stop(false);
	cv::Mat frame;
	cvNamedWindow("show");
	int frame_Num = 0;
	char saveName[256];
	int step = 1;

	while (!stop)
    {
		for (int i = 0; i < step; i++)
		{
			frame_Num++;
			if (!capture.read(frame))
			{
				system("pause");
				return;
			}
		}
		frame = cv::imread("fral001.png");
	Parden:

		found_dHOG.clear();
		found_mHOG.clear();
		found_mC4.clear();

		dHOG.detectMultiScale(frame, found_dHOG, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
		mHOG.detectMultiScale(frame, found_mHOG, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
		found_mC4 = DetectHuman(&IplImage(frame), scanner);

		/*std::vector <CRect> found;
		found.insert(found.end(), found_dHOG.begin(), found_dHOG.end());
		found.insert(found.end(), found_mHOG.begin(), found_mHOG.end());
		found.insert(found.end(), found_mC4.begin(), found_mC4.end());*/
		
		//展示探测到的矩形框
		cv::Mat pic = frame.clone();
		cv::Rect r;
		CRect rc;
		cv::Scalar color = cv::Scalar(255, 0, 0);
		for (int i = 0; i < found_dHOG.size(); i++)
		{
			r = found_dHOG[i];
			rectangle(pic, r.tl(), r.br(), color, 3);
		}
		color = cv::Scalar(0, 255, 0);
		for (int i = 0; i < found_mHOG.size(); i++)
		{
			r = found_mHOG[i];
			rectangle(pic, r.tl(), r.br(), color, 3);
		}
		color = cv::Scalar(0, 0, 255);
		for (int i = 0; i < found_mC4.size(); i++)
		{
			found_mC4[i].Resize(1.05, 1.15);
			rc = found_mC4[i];
			rectangle(pic, cvPoint(rc.left, rc.top), cvPoint(rc.right, rc.bottom), color, 3);
		}
		cv::imshow("show", pic);
		cv::waitKey(1);

		/*if (frame_Num == 967)
		{
			std::cout << "Test" << std::endl;
		}*/
		
		//取交集并做筛选
		std::vector <CRect> found;
		bool contersign;
		double weight1, weight2;
		for (int i = 0; i < found_dHOG.size(); i++)
		{
			for (int j = 0; j < found_mHOG.size(); j++)
			{
				cv::Rect temp1 = found_dHOG[i] & found_mHOG[j];
				weight1 = sqrt(temp1.area()) / sqrt(MAX(found_dHOG[i].area(), found_mHOG[j].area()));
				if (weight1 > 0.9)
				{
					contersign = true;
				}
				else
				{
					contersign = false;
				}
				if (weight1 > 0.7 && temp1.area() > 5000 && temp1.height > 1.8*temp1.width && temp1.height < 4.2 * temp1.width)
				{
					cv::Rect temp2;
					for (int k = 0; k < found_mC4.size(); k++)
					{	
						temp2.x = found_mC4[k].left;
						temp2.y = found_mC4[k].top;
						temp2.width = found_mC4[k].right - found_mC4[k].left;
						temp2.height = found_mC4[k].bottom - found_mC4[k].top;

						cv::Rect temp3 = temp1 & temp2;
						weight2 = sqrt(temp3.area()) / sqrt(temp2.area());
						if (contersign && weight2 > 0.9)
						{
							CRect temp(temp1, (weight1 + weight2) * 1.1);
							temp.Resize(0.95, 0.9);
							found.push_back(temp);
						}
						else if (weight2 > 0.7 && temp3.area() > 5000 && temp3.height > 2.0 * temp3.width && temp3.height < 4.0 * temp3.width)
						{
							CRect temp(temp3, weight1 + weight2);
							found.push_back(temp);
						}
					}
				}
			}
		}
		
		/*if (frame_Num == 687)
		{
			std::cout << "Test" << std::endl;
		}*/
		
		RemoveCoveredRectangles(found, 0.85);

		for (unsigned int i = 0; i < found.size(); i++)
		{
			if (true)//isLogical(found[i], frame))
			{
				CRect rc = found[i];
				rectangle(pic, cvPoint(rc.left, rc.top), cvPoint(rc.right, rc.bottom), CV_RGB(255, 255, 255), 3);
			}
		}
		cv::imshow("show", pic);
		int choose = cvWaitKey(1);
		if (choose == int('p'))
		{
			goto Parden;
		}
		else if (choose == int('l'))
		{
			std::cin >> step;
			std::cout << step << std::endl;
		}
		sprintf_s(saveName, ".\\result\\frame%09d.jpg", frame_Num);
		imwrite(saveName, pic);
		found_save.assign(found.begin(), found.end());
    }
	return;
}
