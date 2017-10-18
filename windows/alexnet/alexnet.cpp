// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<functional>
#include "net.h"
void read_caffe_in(ncnn::Mat &img)
{
	
	img.create(227, 227, 3);
	FILE *fp = fopen("../examples/images/dog227.bin", "rb");
	for (int i = 0; i < img.c;i++)
	{
		float *data = img.data + i*img.cstep;
		fread(data,sizeof(float),img.w*img.h,fp);
		printf("%f\n",data[i]);
	}
	fclose(fp);
	return;
}
static int detect(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet;
    squeezenet.load_param("../examples/alexnet/alexnet.proto");
    squeezenet.load_model("../examples/alexnet/alexnet.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

   // const float mean_vals[3] = {104.f, 117.f, 123.f};
	const float mean_vals[3] = { 104.006989f, 116.668770f, 122.678917f };
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.set_light_mode(false);

#if 1
    ex.input("data", in);
#else
	ncnn::Mat img;
	read_caffe_in(img);
	ex.input("data", img);
#endif
    ncnn::Mat out;
    ex.extract("prob", out);

    cls_scores.resize(out.c);
    for (int j=0; j<out.c; j++)
    {
        const float* prob = out.data + out.cstep * j;
        cls_scores[j] = prob[0];
    }

    return 0;
}

static int print_topk(std::vector<std::string> &vecLabel,const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
		std::string labelname = vecLabel[index];
        fprintf(stderr, "%d %s= %f\n", index, labelname.c_str(),score);
    }

    return 0;
}
void read_label(std::vector<std::string> &vecLabel,std::string label_file)
{
	std::ifstream labels(label_file.c_str());
	if (!labels.is_open())
	{
		fprintf(stderr, "Unable to open labels file %s\n ", label_file.c_str());
	}
	//CHECK(labels) << "Unable to open labels file " << label_file;
	std::string line;
	while (std::getline(labels, line))
		vecLabel.push_back(std::string(line));
}
int main(int argc, char** argv)
{
    //const char* imagepath = argv[1];
	const char* imagepath = "../examples/images/dog.jpg";
	std::vector<std::string> vecLabel;
	std::string label_file = "../examples/images/synset_words.txt";
	read_label(vecLabel,label_file);
    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    detect(m, cls_scores);

    print_topk(vecLabel,cls_scores, 3);
	system("pause");
    return 0;
}

