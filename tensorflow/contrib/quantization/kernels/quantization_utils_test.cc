/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include <limits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/contrib/quantization/kernels/quantization_utils.h"
#include "tensorflow/core/common_runtime/eigen_thread_pool.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class QuantizationUtilsTest : public ::testing::Test {
 protected:
  // If eigen_device is NULL, then the reference implementation is tested.
  void TestRequantizeManyInNewRange32To8Bit(
      Eigen::ThreadPoolDevice* eigen_device) {
    // These are the float values we're going to test the conversions on.
    const size_t values_count = 6;
    const float values[values_count] = {0.0f,  0.45f,  1.0f,
                                        -1.0f, 127.0f, 255.0f};
    // These are the input and output ranges we'll test.
    const size_t ranges_count = 6;
    const float ranges[ranges_count][4] = {
        {0.0f, 255.0f, 0.0f, 255.0f},    //
        {0.0f, 1.0f, 0.0f, 1.0f},        //
        {-1.0f, 1.0f, -1.0f, 1.0f},      //
        {-1.0f, 1.0f, -255.0f, 255.0f},  //
        {3.0f, 3.0f, 0.0f, 255.0f},      // input min == max
        {0.0f, 255.0f, 5.0f, 5.0f},      // output min == max
    };
    for (size_t range_index = 0; range_index < ranges_count; ++range_index) {
      const float input_min = ranges[range_index][0];
      const float input_max = ranges[range_index][1];
      const float output_min = ranges[range_index][2];
      const float output_max = ranges[range_index][3];
      std::vector<qint32> values_quantized;
      std::vector<quint8> expected_values;
      for (size_t value_index = 0; value_index < values_count; ++value_index) {
        const float value_float = values[value_index];
        values_quantized.push_back(
            FloatToQuantized<qint32>(value_float, input_min, input_max));
        expected_values.push_back(FloatToQuantized<quint8>(
            QuantizedToFloat(values_quantized[value_index], input_min,
                             input_max),
            output_min, output_max));
      }

      Tensor i_tensor =
          tensorflow::test::AsTensor(gtl::ArraySlice<qint32>(values_quantized));
      Tensor o_tensor(DT_QUINT8, TensorShape{values_count});
      auto output_values = o_tensor.flat<quint8>();

      if (eigen_device == nullptr) {
        auto input_array = i_tensor.flat<qint32>();
        RequantizeManyInNewRange(input_array.data(), input_array.size(),
                                 input_min, input_max, output_min, output_max,
                                 output_values.data());
      } else {
        RequantizeManyInNewRangeUsingEigen<qint32, quint8>(
            *eigen_device, i_tensor, input_min, input_max, output_min,
            output_max, &o_tensor);
      }

      for (size_t value_index = 0; value_index < values_count; ++value_index) {
        // Here we convert the quantized input value to what we expect
        // to get in the output range.
        ASSERT_EQ(expected_values[value_index], output_values(value_index))
            << "values_quantized[" << value_index
            << "]=" << values_quantized[value_index] << ", values["
            << value_index << "]=" << values[value_index]
            << ", input_min=" << input_min << ", input_max=" << input_max
            << ", output_min=" << output_min << ", output_max=" << output_max
            << ", value_index=" << value_index;
      }
    }
  }

  template <typename InputType, typename OutputType>
  void TestRequantizeManyInNewRangeEigenVsNonEigen() {
    thread::ThreadPool threadpool(Env::Default(), "test", 2 /* num_threads */);
    EigenThreadPoolWrapper wrapper(&threadpool);
    Eigen::ThreadPoolDevice eigen_device(&wrapper, 2 /* num_threads */);

    const size_t ranges_count = 6;
    const float ranges[ranges_count][4] = {
        {0.0f, 255.0f, 0.0f, 255.0f},    //
        {0.0f, 1.0f, 0.0f, 1.0f},        //
        {-1.0f, 1.0f, -1.0f, 1.0f},      //
        {-1.0f, 1.0f, -255.0f, 255.0f},  //
        {3.0f, 3.0f, 0.0f, 255.0f},      // input min == max
        {0.0f, 255.0f, 5.0f, 5.0f},      // output min == max
    };

    // Random values.
    for (size_t range_index = 0; range_index < ranges_count; ++range_index) {
      const float input_min = ranges[range_index][0];
      const float input_max = ranges[range_index][1];
      const float output_min = ranges[range_index][2];
      const float output_max = ranges[range_index][3];
      const int values_count = 10000;
      random::PhiloxRandom philox(testing::RandomSeed(), 17);
      random::SimplePhilox rnd(&philox);
      std::vector<InputType> values_quantized;
      for (int i = 0; i < values_count; ++i) {
        float v = (rnd.RandFloat() * (input_max - input_min)) + input_min;
        values_quantized.push_back(
            FloatToQuantized<InputType>(v, input_min, input_max));
      }

      Tensor i_tensor = tensorflow::test::AsTensor(
          gtl::ArraySlice<InputType>(values_quantized));
      const auto i_array = i_tensor.flat<InputType>();
      Tensor o_tensor_eigen(DataTypeToEnum<OutputType>::v(),
                            TensorShape{values_count});
      auto output_values_eigen = o_tensor_eigen.flat<OutputType>();
      Tensor o_tensor_ref(DataTypeToEnum<OutputType>::v(),
                          TensorShape{values_count});
      auto output_values_ref = o_tensor_ref.flat<OutputType>();

      RequantizeManyInNewRange(i_array.data(), i_array.size(), input_min,
                               input_max, output_min, output_max,
                               output_values_ref.data());
      RequantizeManyInNewRangeUsingEigen<InputType, OutputType>(
          eigen_device, i_tensor, input_min, input_max, output_min, output_max,
          &o_tensor_eigen);

      for (int i = 0; i < values_quantized.size(); ++i) {
        EXPECT_EQ(output_values_eigen(i), output_values_ref(i)) << i;
      }
    }
  }
};

TEST_F(QuantizationUtilsTest, FloatToQuantized) {
  EXPECT_EQ(quint8(0), FloatToQuantized<quint8>(0.0f, 0.0f, 1.0f));
  EXPECT_EQ(quint8(0), FloatToQuantized<quint8>(0.0f, 0.0f, 2.0f));
  EXPECT_EQ(quint8(128), FloatToQuantized<quint8>(0.5f, 0.0f, 1.0f));
  EXPECT_EQ(quint8(128), FloatToQuantized<quint8>(1.0f, 0.0f, 2.0f));
  EXPECT_EQ(quint8(255), FloatToQuantized<quint8>(1.0f, 0.0f, 1.0f));
  EXPECT_EQ(quint8(255), FloatToQuantized<quint8>(2.0f, 0.0f, 2.0f));
  EXPECT_EQ(quint8(0), FloatToQuantized<quint8>(-128.0f, -128.0f, 127.0f));
  EXPECT_EQ(quint8(128), FloatToQuantized<quint8>(0.0f, -128.0f, 127.0f));
  EXPECT_EQ(quint8(255), FloatToQuantized<quint8>(127.0f, -128.0f, 127.0f));
  EXPECT_EQ(quint8(0), FloatToQuantized<quint8>(1.0f, 1.0f, 256.0f));
  EXPECT_EQ(quint8(127), FloatToQuantized<quint8>(128.0f, 1.0f, 256.0f));
  EXPECT_EQ(quint8(255), FloatToQuantized<quint8>(256.0f, 1.0f, 256.0f));

  const int int32_min = std::numeric_limits<int>::min();
  const int int32_max = std::numeric_limits<int>::max();

  EXPECT_EQ(qint32(int32_min),
            FloatToQuantized<qint32>(-128.0f, -128.0f, 128.0f));
  EXPECT_EQ(qint32(0), FloatToQuantized<qint32>(0.0f, -128.0f, 128.0f));
  EXPECT_EQ(qint32(int32_max),
            FloatToQuantized<qint32>(128.0f, -128.0f, 128.0f));
}

TEST_F(QuantizationUtilsTest, QuantizedToFloat) {
  EXPECT_LT(fabsf(0.0f - QuantizedToFloat<quint8>(0, 0.0f, 1.0f)), 1 / 255.0f);
  EXPECT_LT(fabsf(0.0f - QuantizedToFloat<quint8>(0, 0.0f, 2.0f)), 1 / 255.0f);
  EXPECT_LT(fabsf(0.5f - QuantizedToFloat<quint8>(127, 0.0f, 1.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(1.0f - QuantizedToFloat<quint8>(127, 0.0f, 2.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(1.0f - QuantizedToFloat<quint8>(255, 0.0f, 1.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(2.0f - QuantizedToFloat<quint8>(255, 0.0f, 2.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(1.0f - QuantizedToFloat<quint8>(0, 1.0f, 256.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(128.0f - QuantizedToFloat<quint8>(127, 1.0f, 256.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(256.0f - QuantizedToFloat<quint8>(255, 1.0f, 256.0f)),
            1 / 255.0f);

  const int int32_min = std::numeric_limits<int>::min();
  const int int32_max = std::numeric_limits<int>::max();

  EXPECT_LT(
      fabsf(-1.0f - QuantizedToFloat<qint32>(qint32(int32_min), -1.0f, 1.0f)),
      1e-5f);
  EXPECT_LT(fabsf(0.0f - QuantizedToFloat<qint32>(qint32(0), -1.0f, 1.0f)),
            1e-5f);
  EXPECT_LT(
      fabsf(1.0f - QuantizedToFloat<qint32>(qint32(int32_max), -1.0f, 1.0f)),
      1e-5f);
}

TEST_F(QuantizationUtilsTest, AvoidBias) {
  for (int i = 0; i < 256; ++i) {
    const float as_float = QuantizedToFloat<quint8>(i, 0.0f, 2.0f);
    const int back_to_int = FloatToQuantized<quint8>(as_float, 0.0f, 2.0f);
    EXPECT_EQ(i, back_to_int);
  }
}

TEST_F(QuantizationUtilsTest, RequantizeInNewRange) {
  // These are the float values we're going to test the conversions on.
  const size_t values_count = 6;
  const float values[values_count] = {0.0f, 0.5f, 1.0f, -1.0f, 127.0f, 255.0f};
  // These are the input and output ranges we'll test.
  const size_t ranges_count = 4;
  const float ranges[ranges_count][4] = {
      {0.0f, 255.0f, 0.0f, 255.0f},
      {0.0f, 1.0f, 0.0f, 1.0f},
      {-1.0f, 1.0f, -1.0f, 1.0f},
      {-1.0f, 1.0f, -255.0f, 255.0f},
  };
  for (size_t value_index = 0; value_index < values_count; ++value_index) {
    const float value_float = values[value_index];
    for (size_t range_index = 0; range_index < ranges_count; ++range_index) {
      const float input_min = ranges[range_index][0];
      const float input_max = ranges[range_index][1];
      const float output_min = ranges[range_index][2];
      const float output_max = ranges[range_index][3];
      const quint8 input_value =
          FloatToQuantized<quint8>(value_float, input_min, input_max);
      // Here we convert the quantized input value to what we expect
      // to get in the output range.
      const qint32 expected_value = FloatToQuantized<qint32>(
          QuantizedToFloat(input_value, input_min, input_max), output_min,
          output_max);
      EXPECT_EQ(expected_value,
                (RequantizeInNewRange<quint8, qint32>(
                    input_value, input_min, input_max, output_min, output_max)))
          << "value_float=" << value_float << ", input_min=" << input_min
          << ", input_max=" << input_max << ", output_min=" << output_min
          << ", output_max=" << output_max;
    }
  }
}

TEST_F(QuantizationUtilsTest, RequantizeInNewRangeRealData) {
  const float value_as_float = -0.290169f;
  const float input_min = -0.739539f;
  const float input_max = 0.641057f;
  const float output_min = -2381.49f;
  const float output_max = 2207.6f;
  const quint8 value_as_quint8 =
      FloatToQuantized<quint8>(value_as_float, input_min, input_max);
  EXPECT_EQ(quint8(83), value_as_quint8);
  const qint32 actual_output = RequantizeInNewRange<quint8, qint32>(
      value_as_quint8, input_min, input_max, output_min, output_max);
  const qint32 value_as_qint32 =
      FloatToQuantized<qint32>(value_as_float, output_min, output_max);
  EXPECT_LT(std::abs(value_as_qint32 - actual_output), 10);
}

TEST_F(QuantizationUtilsTest, RequantizeInNewRange32To8Bit) {
  // These are the float values we're going to test the conversions on.
  const size_t values_count = 6;
  const float values[values_count] = {0.0f, 0.45f, 1.0f, -1.0f, 127.0f, 255.0f};
  // These are the input and output ranges we'll test.
  const size_t ranges_count = 4;
  const float ranges[ranges_count][4] = {
      {0.0f, 255.0f, 0.0f, 255.0f},
      {0.0f, 1.0f, 0.0f, 1.0f},
      {-1.0f, 1.0f, -1.0f, 1.0f},
      {-1.0f, 1.0f, -255.0f, 255.0f},
  };
  for (size_t value_index = 0; value_index < values_count; ++value_index) {
    const float value_float = values[value_index];
    for (size_t range_index = 0; range_index < ranges_count; ++range_index) {
      const float input_min = ranges[range_index][0];
      const float input_max = ranges[range_index][1];
      const float output_min = ranges[range_index][2];
      const float output_max = ranges[range_index][3];
      const qint32 input_value =
          FloatToQuantized<qint32>(value_float, input_min, input_max);
      // Here we convert the quantized input value to what we expect
      // to get in the output range.
      const quint8 expected_value = FloatToQuantized<quint8>(
          QuantizedToFloat(input_value, input_min, input_max), output_min,
          output_max);
      EXPECT_EQ(expected_value,
                (RequantizeInNewRange<qint32, quint8>(
                    input_value, input_min, input_max, output_min, output_max)))
          << "input_value=" << input_value << ", value_float=" << value_float
          << ", input_min=" << input_min << ", input_max=" << input_max
          << ", output_min=" << output_min << ", output_max=" << output_max;
    }
  }
}

TEST_F(QuantizationUtilsTest, RequantizeManyInNewRange32To8Bit) {
  TestRequantizeManyInNewRange32To8Bit(nullptr /* eigen_device */);
}

#if 0
TEST_F(QuantizationUtilsTest, RequantizeManyInNewRange32To8BitUsingEigen) {
  thread::ThreadPool threadpool(Env::Default(), "test", 2 /* num_threads */);
  EigenThreadPoolWrapper wrapper(&threadpool);
  Eigen::ThreadPoolDevice eigen_device(&wrapper, 2 /* num_threads */);
  TestRequantizeManyInNewRange32To8Bit(&eigen_device);
}
#endif

TEST_F(QuantizationUtilsTest, RequantizeManyInNewRange32To8BitEigenVsNonEigen) {
  TestRequantizeManyInNewRangeEigenVsNonEigen<qint32, quint8>();
}

TEST_F(QuantizationUtilsTest,
       RequantizeManyInNewRange32To8BitSignedEigenVsNonEigen) {
  TestRequantizeManyInNewRangeEigenVsNonEigen<qint32, qint8>();
}

TEST_F(QuantizationUtilsTest, FloatTensorToQuantized) {
  const int input_width = 3;
  const int input_height = 3;
  const float input_min = 0.0f;
  const float input_max = 255.0f;
  Tensor input(DT_FLOAT, TensorShape({input_height, input_width}));
  test::FillValues<float>(&input, {1.0f, -1.0f, 10.0f, 10.25f, 127.0f, 255.0f,
                                   512.0f, 0.0f, 23.0f});
  Tensor expected(DT_QUINT8, TensorShape({input_height, input_width}));
  test::FillValues<quint8>(&expected, {1, 0, 10, 10, 127, 255, 255, 0, 23});
  Tensor output = FloatTensorToQuantized<quint8>(input, input_min, input_max);
  test::ExpectTensorEqual<quint8>(expected, output);
}

TEST_F(QuantizationUtilsTest, QuantizedTensorToFloat) {
  const int input_width = 3;
  const int input_height = 3;
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  Tensor input(DT_QUINT8, TensorShape({input_height, input_width}));
  test::FillValues<quint8>(&input, {0, 128, 255, 23, 24, 25, 243, 244, 245});
  Tensor expected(DT_FLOAT, TensorShape({input_height, input_width}));
  test::FillValues<float>(&expected, {-128.0f, 0.0f, 127.0f, -105.0f, -104.0f,
                                      -103.0f, 115.0f, 116.0f, 117.0f});
  Tensor output = QuantizedTensorToFloat<quint8>(input, input_min, input_max);
  test::ExpectTensorEqual<float>(expected, output);
}

}  // namespace tensorflow
