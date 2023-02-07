#include "SPextractor.h"


using namespace cv;
using namespace std;

namespace ORB_SLAM2
{
    SPextractor::SPextractor(const std::string& modelPath, int _nfeatures, float _scaleFactor, int _nlevels):
            nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels)
    {
        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0]=1.0f;
        mvLevelSigma2[0]=1.0f;
        for(int i=1; i<nlevels; i++)
        {
            mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
            mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSigma2.resize(nlevels);
        for(int i=0; i<nlevels; i++)
        {
            mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
            mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
        }

        mnFeaturesPerLevel.resize(nlevels);
        float factor = 1.0f / scaleFactor;
        float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

        int sumFeatures = 0;
        for( int level = 0; level < nlevels-1; level++ )
        {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

        // new for super point
        std::cout << "SuperPointExtractor:: " << modelPath << "\n";
        device = getDevice();
        model = loadModel(modelPath, device);
    }


    void SPextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                                  cv::OutputArray _descriptors)
    {
        if(_image.empty())
            return;

        Mat image = _image.getMat();
        image.convertTo(image, CV_32FC1, 1.0 / 255.0f, 0);
        auto s = std::chrono::system_clock::now();
        model->eval();

        auto i_tensor = matToTensor(image);
        i_tensor = prepareTensor(i_tensor, device);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(i_tensor);

        auto outputs = model->forward(inputs).toTuple();

        auto semi = outputs->elements()[0].toTensor().to(torch::kCPU);
        auto desc = outputs->elements()[1].toTensor().to(torch::kCPU);
//
        computeKeypoints(image.size(), semi, _keypoints);

        Mat descriptors;
//        _descriptors.create(_keypoints.size(), 32, CV_8U);
        _descriptors.create(_keypoints.size(), 256, CV_32FC1);
        descriptors = _descriptors.getMat();
        Mat Desc = descriptors.rowRange(0, _keypoints.size());

        computeDescriptors(desc, _keypoints, image.size(), Desc);

        auto e = std::chrono::system_clock::now();
        std::cout<<" cost: "<<std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()<<" ms"<<std::endl;
    }

    torch::DeviceType SPextractor::getDevice()
    {
        std::cout << "SuperPointExtractor:: " << "getDevice()" << "\n";

        torch::DeviceType device_type;
        if (torch::cuda::is_available()) {
            std::cout << "SuperPointExtractor:: " << "CUDA available! Training on GPU." << std::endl;
            device_type = torch::kCUDA;
        }
        else {
            std::cout << "SuperPointExtractor:: " << "Training on CPU." << std::endl;
            device_type = torch::kCPU;
        }
        return device_type;
    }

    std::shared_ptr<torch::jit::script::Module> SPextractor::loadModel(const string &path, torch::Device device)
    {
        std::cout << "SuperPointExtractor:: " << "loadModel" << "\n";
        std::shared_ptr<torch::jit::script::Module> module;
        try {
            module = std::make_shared<torch::jit::script::Module>(torch::jit::load(path, device));
        }
        catch (const c10::Error& e) {
            std::cerr << "SuperPointExtractor:: " << "error loading the model\n";
        }
        return std::move(module);
    }

    void SPextractor::nms_fast(int h, int w, std::vector<cv::KeyPoint> points, vector <cv::KeyPoint> &nms_points,
                               int nms_distance)
    {
        using std::vector;
        using std::tuple;
        nms_points.clear();
        vector<vector<int>> grid(h, vector<int>(w, 0));

        // sort as per scores
        std::sort(points.begin(), points.end(), [](const cv::KeyPoint& t1, const cv::KeyPoint& t2) -> bool {
            return t1.response > t2.response;
        });

        /*
            initialize grid
            -1	= kept
            0	= supressed
            1	= unvisited
        */
        for (auto & point : points) {
            grid[(int)point.pt.y][(int)point.pt.x] = 1;
        }

        int suppressed_points = 0;

        for (auto & point : points) {
            int row = (int)point.pt.y;
            int col = (int)point.pt.x;
            float val = point.response;
            /*
                supress border points by default
            */
            if (row > nms_distance && row < h - nms_distance && col > nms_distance && col < w - nms_distance) {
                if (grid[row][col] == 1) {

                    for (int k_row = -nms_distance / 2; k_row <= nms_distance / 2; k_row++) {
                        for (int k_col = -nms_distance / 2; k_col <= nms_distance / 2; k_col++) {
                            grid[row + k_row][col + k_col] = 0;
                        }
                    }
                    grid[row][col] = -1;
                    suppressed_points++;
                    nms_points.emplace_back(cv::Point2f(col, row),0,0,val);
                }
            }
        }
    }

    void SPextractor::computeKeypoints(const Size &img, torch::Tensor &semi, vector <cv::KeyPoint> &cv_kp)
    {
        std::vector<cv::KeyPoint> points;

        auto dense	= torch::softmax(semi, 1);

        auto nodust = dense.slice(1, 0, 64);
        nodust = nodust.permute({ 0, 2, 3, 1 });

        int H = nodust.size(1);
        int W = nodust.size(2);

        semi = nodust.contiguous().view({-1, H, W, 8, 8});
        semi = semi.permute({0, 1, 3, 2, 4});
        auto heatmap = semi.contiguous().view({-1, H*8, W*8});

        heatmap = heatmap.squeeze(0);

        auto yx_idx = heatmap > 0.015f;

        yx_idx = torch::nonzero(yx_idx);

        auto rows = yx_idx.select(1, 0);
        auto cols = yx_idx.select(1, 1);

        for (int i = 0; i < yx_idx.size(0); i++) {
            int row = rows[i].item<int>();
            int col = cols[i].item<int>();

            points.emplace_back(cv::Point2f(col, row),0,0,heatmap[row][col].item<float>());
        }
//        std::cout<<" points: "<<points.size()<<std::endl;
//        if (nfeatures > 3000){
        nms_fast(heatmap.size(0), heatmap.size(1), std::move(points), cv_kp, 1);
//        }else{
//            nms_fast(heatmap.size(0), heatmap.size(1), std::move(points), cv_kp, 2);
//        }

        std::cout<<" nms: "<<cv_kp.size()<<std::endl;
    }

    void
    SPextractor::computeDescriptors(const torch::Tensor &pred_desc, const vector <cv::KeyPoint> &keypoints, const Size &size,
                                    Mat &cv_desc)
    {
        // Descriptors
        int D = pred_desc.size(1);
        auto sample_pts = torch::zeros({ static_cast<int>(keypoints.size()), 2});
        for (int i = 0; i < keypoints.size(); i++) {
            sample_pts[i][0] = keypoints.at(i).pt.y;
            sample_pts[i][1] = keypoints.at(i).pt.x;
        }
        sample_pts = sample_pts.to(torch::kFloat);

        auto grid = torch::zeros({ 1, 1, sample_pts.size(0), 2 });

        /*
            z-score points for grid zampler
        */
        grid[0][0].slice(1, 0, 1) = sample_pts.slice(1, 1, 2) / (float(size.width) - 1.0f); // xs
        grid[0][0].slice(1, 1, 2) = sample_pts.slice(1, 0, 1) / (float(size.height) - 1.0f); // ys

        auto desc = torch::grid_sampler(pred_desc, grid, 0, 0, false);
        desc = desc.squeeze(0).squeeze(1);

        auto dn = torch::norm(desc, 2, 1);
        desc = desc.div(torch::unsqueeze(dn, 1));

        desc = desc.transpose(0, 1).contiguous();

//        cv_desc.create((int)keypoints.size(), 256, CV_32FC1);
        memcpy((void*)cv_desc.data, desc.data_ptr(), sizeof(float) * desc.numel());//sizeof(float) * desc.numel()

    }


    torch::Tensor SPextractor::matToTensor(const Mat &image)
    {
        std::vector<int64_t> dims = { 1, image.rows, image.cols, image.channels() };
        return torch::from_blob(
                image.data, dims, torch::kFloat
        );
    }

    torch::Tensor SPextractor::prepareTensor(const torch::Tensor &tensor, const torch::Device &device)
    {
        torch::Tensor a = tensor.permute({ 0, 3, 1, 2 });
        a = a.to(device);
        return std::move(a);
    }
} //namespace SP_SLAM
