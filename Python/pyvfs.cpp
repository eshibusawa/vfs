// BSD 2-Clause License
//
// Copyright (c) 2022, Eijiro SHIBUSAWA
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stereotgv/stereotgv.h>
#include <opencv2/core/core.hpp>

#include <iostream>

namespace py = pybind11;

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
		const int stereoWidth = obj.attr("width").cast<int>();
		const int stereoHeight = obj.attr("height").cast<int>();
		const float beta = obj.attr("beta").cast<float>();
		const float gamma = obj.attr("gamma").cast<float>();
		const float alpha0 = obj.attr("alpha0").cast<float>();
		const float alpha1 = obj.attr("alpha1").cast<float>();
		const float timeStepLambda = obj.attr("timeStepLambda").cast<float>();
		const float lambda = obj.attr("Lambda").cast<float>();
		const int nLevel = obj.attr("nLevel").cast<int>();
		const float fScale = obj.attr("fScale").cast<float>();
		const int nWarpIters = obj.attr("nWarpIters").cast<int>();
		const int nSolverIters = obj.attr("nSolverIters").cast<int>();
		const float limitRange = obj.attr("limitRange").cast<float>();
		m_stereotgv->limitRange = limitRange;
		int ret = m_stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
			timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
		m_stereotgv->visualizeResults = true;
		m_initialized = true;
		m_stereoWidth = stereoWidth;
		m_stereoHeight = stereoHeight;
		return ret == 0;
	}

	bool copyMaskToDevice(const py::array &mask)
	{
		py::buffer_info buf = mask.request();
		if ((buf.ndim != 2) || (buf.strides[0] != buf.shape[1] * sizeof(float)) ||
			(buf.format != py::format_descriptor<float>::format()))
		{
			std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
			return false;
		}

		cv::Mat fisheyeMask(buf.shape[0], buf.shape[1], CV_32F, buf.ptr);
		int ret = m_stereotgv->copyMaskToDevice(fisheyeMask);
		return ret == 0;
	}

	bool loadVectorFields(const py::array &translationArr, const py::array &calibrationArr)
	{
		py::buffer_info buf1 = translationArr.request();
		py::buffer_info buf2 = calibrationArr.request();
		if ((buf1.ndim != 3) || (buf1.strides[0] != 2 * buf1.shape[1] * sizeof(float)) ||
			(buf1.format != py::format_descriptor<float>::format()) ||
			(buf2.ndim != 3) || (buf2.strides[0] != 2 * buf2.shape[1] * sizeof(float)) ||
			(buf2.format != buf1.format)
			)
		{
			std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
			return false;
		}

		cv::Mat translationVector(buf1.shape[0], buf1.shape[1], CV_32FC2, buf1.ptr);
		cv::Mat calibrationVector(buf2.shape[0], buf2.shape[1], CV_32FC2, buf2.ptr);
		int ret = m_stereotgv->loadVectorFields(translationVector, calibrationVector);
		return ret == 0;
	}

	bool copyImagesToDevice(const py::array &arr1, const py::array &arr2)
	{
		py::buffer_info buf1 = arr1.request();
		py::buffer_info buf2 = arr2.request();
		if ((buf1.ndim != 2) || (buf1.strides[0] != buf1.shape[1] * sizeof(unsigned char)) ||
			(buf1.format != py::format_descriptor<unsigned char>::format()) ||
			(buf2.ndim != 2) || (buf2.strides[0] != buf2.shape[1] * sizeof(unsigned char)) ||
			(buf2.format != buf1.format)
			)
		{
			std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
			return false;
		}

		cv::Mat mat1(buf1.shape[0], buf1.shape[1], CV_8U, buf1.ptr);
		cv::Mat mat2(buf2.shape[0], buf2.shape[1], CV_8U, buf2.ptr);
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
		const float focalx = obj.attr("focalx").cast<float>();
		const float focaly = obj.attr("focaly").cast<float>();
		const float cx = obj.attr("cx").cast<float>();
		const float cy = obj.attr("cy").cast<float>();
		const float d0 = obj.attr("d0").cast<float>();
		const float d1 = obj.attr("d1").cast<float>();
		const float d2 = obj.attr("d2").cast<float>();
		const float d3 = obj.attr("d3").cast<float>();
		const float tx = obj.attr("tx").cast<float>();
		const float ty = obj.attr("ty").cast<float>();
		const float tz = obj.attr("tz").cast<float>();

		cv::Mat depth = cv::Mat(m_stereoHeight, m_stereoWidth, CV_32F);
		cv::Mat X = cv::Mat(m_stereoHeight, m_stereoWidth, CV_32FC3);
		int ret = m_stereotgv->copyStereoToHost(depth, X, focalx, focaly,
			cx, cy,
			d0, d1, d2, d3,
			tx, ty, tz);
		if (ret != 0)
		{
			std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
			return py::none();
		}

		cv::Mat disparity = cv::Mat(m_stereoHeight, m_stereoWidth, CV_32FC2);
		ret = m_stereotgv->copyDisparityToHost(disparity);
		if (ret != 0)
		{
			std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
			return py::none();
		}

		cv::Mat uvrgb = cv::Mat(m_stereoHeight, m_stereoWidth, CV_32FC3);
		ret = m_stereotgv->copyDisparityVisToHost(uvrgb, 40.0f);
		if (ret != 0)
		{
			std::cerr << __FILE__ << " : " << __LINE__ << std::endl;
			return py::none();
		}

		// copy depth
		auto depthArr = py::array_t<float, py::array::c_style>({depth.rows, depth.cols});
		py::buffer_info depthBuf = depthArr.request();
		float *pDepthArr = reinterpret_cast<float *>(depthBuf.ptr);
		depth.copyTo(cv::Mat(depth.rows, depth.cols, CV_32F, pDepthArr));
		// copy X
		auto xArr = py::array_t<float, py::array::c_style>({X.rows, X.cols, 3});
		py::buffer_info xBuf = xArr.request();
		float *pXArr = reinterpret_cast<float *>(xBuf.ptr);
		X.copyTo(cv::Mat(X.rows, X.cols, CV_32FC3, pXArr));
		// copy disparity
		auto disparityArr = py::array_t<float, py::array::c_style>({disparity.rows, disparity.cols, 2});
		py::buffer_info disparityBuf = disparityArr.request();
		float *pDisparityArr = reinterpret_cast<float *>(disparityBuf.ptr);
		disparity.copyTo(cv::Mat(disparity.rows, disparity.cols, CV_32FC2, pDisparityArr));
		// copy UV RGB
		auto uvrgbArr = py::array_t<float, py::array::c_style>({uvrgb.rows, uvrgb.cols, 3});
		py::buffer_info uvrgbBuf = uvrgbArr.request();
		float *pUvrgbArr = reinterpret_cast<float *>(uvrgbBuf.ptr);
		uvrgb.copyTo(cv::Mat(uvrgb.rows, uvrgb.cols, CV_32FC3, pUvrgbArr));

		return py::make_tuple(depthArr, xArr, disparityArr, uvrgbArr);
	}

private:
	StereoTgv *m_stereotgv;
	bool m_initialized;
	int m_stereoWidth;
	int m_stereoHeight;
};

PYBIND11_MODULE(pyvfs, m)
{
	py::class_<VFS> vfs(m, "vfs");

	vfs.def(py::init<>())
		.def("initialize", &VFS::initialize)
		.def("copy_mask_to_device", &VFS::copyMaskToDevice)
		.def("load_vector_fields", &VFS::loadVectorFields)
		.def("copy_images_to_device", &VFS::copyImagesToDevice)
		.def("solve_stereo_forward_masked", &VFS::solveStereoForwardMasked)
		.def("copy_result_to_host", &VFS::copyResultToHost)
		;
}
