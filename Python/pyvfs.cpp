#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <stereotgv/stereotgv.h>
#include <opencv2/core/core.hpp>

#include <iostream>

namespace py = boost::python;
namespace np = boost::python::numpy;

class VFS
{
public:
	VFS():
		m_stereotgv(new StereoTgv())
		, m_initialized(false)
		, m_stereoWidth(0)
		, m_stereoHeight(0)
	{
	}

	~VFS()
	{
		delete m_stereotgv;
	}

	bool initialize(const py::object &obj)
	{
		if (m_initialized)
		{
			std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
			return false;
		}
		const int stereoWidth = py::extract<int>(obj.attr("width"));
		const int stereoHeight = py::extract<int>(obj.attr("height"));
		const float beta = py::extract<float>(obj.attr("beta"));
		const float gamma = py::extract<float>(obj.attr("gamma"));
		const float alpha0 = py::extract<float>(obj.attr("alpha0"));
		const float alpha1 = py::extract<float>(obj.attr("alpha1"));
		const float timeStepLambda = py::extract<float>(obj.attr("timeStepLambda"));
		const float lambda = py::extract<float>(obj.attr("Lambda"));
		const int nLevel = py::extract<int>(obj.attr("nLevel"));
		const float fScale = py::extract<float>(obj.attr("fScale"));
		const int nWarpIters = py::extract<int>(obj.attr("nWarpIters"));
		const int nSolverIters = py::extract<int>(obj.attr("nSolverIters"));
		const float limitRange = py::extract<float>(obj.attr("limitRange"));
		m_stereotgv->limitRange = limitRange;
		int ret = m_stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
			timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
		m_stereotgv->visualizeResults = true;
		m_initialized = true;
		m_stereoWidth = stereoWidth;
		m_stereoHeight = stereoHeight;
		return ret == 0;
	}

	bool copyMaskToDevice(const np::ndarray &mask)
	{
		const int nd = mask.get_nd();
		auto ps = mask.get_shape();
		if ((nd != 2) || (!(mask.get_flags() & np::ndarray::C_CONTIGUOUS)) ||
				(mask.get_dtype() != np::dtype::get_builtin<float>()))
		{
			std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
			return false;
		}

		cv::Mat fisheyeMask(ps[0], ps[1], CV_32F, mask.get_data());
		int ret = m_stereotgv->copyMaskToDevice(fisheyeMask);
		return ret == 0;
	}

	bool loadVectorFields(const np::ndarray &translationArr, const np::ndarray &calibrationArr)
	{
		const int nd1 = translationArr.get_nd();
		auto ps1 = translationArr.get_shape();
		const int nd2 = calibrationArr.get_nd();
		auto ps2 = calibrationArr.get_shape();

		if ((nd1 != 3) || (!(translationArr.get_flags() & np::ndarray::C_CONTIGUOUS)) ||
				(translationArr.get_dtype() != np::dtype::get_builtin<float>()) ||
			(nd2 != 3) || (!(calibrationArr.get_flags() & np::ndarray::C_CONTIGUOUS)) ||
							(calibrationArr.get_dtype() != np::dtype::get_builtin<float>()))
		{
			std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
			return false;
		}

		cv::Mat translationVector(ps1[0], ps1[1], CV_32FC2, translationArr.get_data());
		cv::Mat calibrationVector(ps2[0], ps2[1], CV_32FC2, calibrationArr.get_data());
		int ret = m_stereotgv->loadVectorFields(translationVector, calibrationVector);
		return ret == 0;
	}

	bool copyImagesToDevice(const np::ndarray &arr1, const np::ndarray &arr2)
	{
		const int nd1 = arr1.get_nd();
		auto ps1 = arr1.get_shape();
		const int nd2 = arr2.get_nd();
		auto ps2 = arr2.get_shape();

		if ((nd1 != 2) || (!(arr1.get_flags() & np::ndarray::C_CONTIGUOUS)) ||
				(arr1.get_dtype() != np::dtype::get_builtin<unsigned char>()) ||
			(nd2 != 2) || (!(arr2.get_flags() & np::ndarray::C_CONTIGUOUS)) ||
							(arr2.get_dtype() != np::dtype::get_builtin<unsigned char>()))
		{
				std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
				return false;
		}

		cv::Mat mat1(ps1[0], ps1[1], CV_8U, arr1.get_data());
		cv::Mat mat2(ps2[0], ps2[1], CV_8U, arr2.get_data());
		int ret = m_stereotgv->copyImagesToDevice(mat1, mat2);
		return ret == 0;
	}

	bool solveStereoForwardMasked()
	{
		clock_t start = clock();
		int ret = m_stereotgv->solveStereoForwardMasked();
		clock_t timeElapsed = (clock() - start);
		std::cout << "time: " << timeElapsed << " ms ";
		if (ret != 0)
		{
			std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
			return false;
		}

		return true;
	}

	py::object copyResultToHost(const py::object &obj)
	{
		const float focalx = py::extract<float>(obj.attr("focalx"));
		const float focaly = py::extract<float>(obj.attr("focaly"));
		const float cx = py::extract<float>(obj.attr("cx"));
		const float cy = py::extract<float>(obj.attr("cy"));
		const float d0 = py::extract<float>(obj.attr("d0"));
		const float d1 = py::extract<float>(obj.attr("d1"));
		const float d2 = py::extract<float>(obj.attr("d2"));
		const float d3 = py::extract<float>(obj.attr("d3"));
		const float tx = py::extract<float>(obj.attr("tx"));
		const float ty = py::extract<float>(obj.attr("ty"));
		const float tz = py::extract<float>(obj.attr("tz"));

		cv::Mat depth = cv::Mat(m_stereoHeight, m_stereoWidth, CV_32F);
		cv::Mat X = cv::Mat(m_stereoHeight, m_stereoWidth, CV_32FC3);
		int ret = m_stereotgv->copyStereoToHost(depth, X, focalx, focaly,
			cx, cy,
			d0, d1, d2, d3,
			tx, ty, tz);
		if (ret != 0)
		{
			std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
			return py::object();
		}

		cv::Mat disparity = cv::Mat(m_stereoHeight, m_stereoWidth, CV_32FC2);
		ret = m_stereotgv->copyDisparityToHost(disparity);
		if (ret != 0)
		{
			std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
			return py::object();
		}

		cv::Mat uvrgb = cv::Mat(m_stereoHeight, m_stereoWidth, CV_32FC3);
		ret = m_stereotgv->copyDisparityVisToHost(uvrgb, 40.0f);
		if (ret != 0)
		{
			std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
			return py::object();
		}

		// copy depth
		np::ndarray depthArr = np::empty(py::make_tuple(depth.rows, depth.cols), np::dtype::get_builtin<float>());
		float *pDepthArr = reinterpret_cast<float *>(depthArr.get_data());
		depth.copyTo(cv::Mat(depth.rows, depth.cols, CV_32F, pDepthArr));
		// copy X
		np::ndarray xArr = np::empty(py::make_tuple(X.rows, X.cols, 3), np::dtype::get_builtin<float>());
		float *pXArr = reinterpret_cast<float *>(xArr.get_data());
		X.copyTo(cv::Mat(X.rows, X.cols, CV_32FC3, pXArr));
		// copy disparity
		np::ndarray disparityArr = np::empty(py::make_tuple(disparity.rows, disparity.cols, 2), np::dtype::get_builtin<float>());
		float *pDisparityArr = reinterpret_cast<float *>(disparityArr.get_data());
		disparity.copyTo(cv::Mat(disparity.rows, disparity.cols, CV_32FC2, pDisparityArr));
		// copy UV RGB
		np::ndarray uvrgbArr = np::empty(py::make_tuple(uvrgb.rows, uvrgb.cols, 3), np::dtype::get_builtin<float>());
		float *pUvrgbArr = reinterpret_cast<float *>(uvrgbArr.get_data());
		uvrgb.copyTo(cv::Mat(uvrgb.rows, uvrgb.cols, CV_32FC3, pUvrgbArr));

		return py::make_tuple(depthArr, xArr, disparityArr, uvrgbArr);
	}

private:
	StereoTgv *m_stereotgv;
	bool m_initialized;
	int m_stereoWidth;
	int m_stereoHeight;
};

BOOST_PYTHON_MODULE(pyvfs)
{
	Py_Initialize();
	np::initialize();
	py::class_<VFS>("vfs")
		.def("initialize", &VFS::initialize)
		.def("copy_mask_to_device", &VFS::copyMaskToDevice)
		.def("load_vector_fields", &VFS::loadVectorFields)
		.def("copy_images_to_device", &VFS::copyImagesToDevice)
		.def("solve_stereo_forward_masked", &VFS::solveStereoForwardMasked)
		.def("copy_result_to_host", &VFS::copyResultToHost)
	;
}
