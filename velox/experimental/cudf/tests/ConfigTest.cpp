      {CudfConfig::kCudfAllowCpuFallback, "false"},
      {CudfConfig::kUcxExchange, "true"},
      {CudfConfig::kUcxxErrorHandling, "false"},
      {CudfConfig::kUcxIntraNodeExchange, "true"},
      {CudfConfig::kUcxxBlockingPolling, "false"},
      {CudfConfig::kUcxExchangeLogLevel, "2"},
      {CudfConfig::kUcxPartitionedOutputBatchRows, "100000"},
      {CudfConfig::kUcxExchangeCompression, "column-adaptive-freq-pfor-min128"},
      {CudfConfig::kUcxExchangeCompressionPipeline, "true"},
      {CudfConfig::kUcxExchangeCompressionPipelineThreads, "2"},
      {CudfConfig::kUcxExchangeCompressionMinBytes, "268435456"},
      {CudfConfig::kUcxExchangeCompressionSafetyMargin, "1.5"}};

  CudfConfig config;
  ASSERT_FALSE(config.streamingGroupbyEnabled);
  ASSERT_EQ(config.streamingGroupbyCapacityMultiplier, 2.0);
  config.initialize(std::move(options));
  ASSERT_EQ(config.enabled, false);
  ASSERT_EQ(config.debugEnabled, true);
  ASSERT_EQ(config.memoryResource, "arena");
  ASSERT_EQ(config.memoryPercent, 25);
  ASSERT_EQ(config.functionNamePrefix, "presto");
  ASSERT_EQ(config.streamingGroupbyEnabled, true);
  ASSERT_EQ(config.streamingGroupbyCapacityMultiplier, 3.5);
  ASSERT_EQ(config.allowCpuFallback, false);
  ASSERT_TRUE(config.exchange);
  ASSERT_FALSE(config.ucxxErrorHandling);
  ASSERT_TRUE(config.intraNodeExchange);
  ASSERT_FALSE(config.ucxxBlockingPolling);
  ASSERT_EQ(config.exchangeLogLevel, 2);
  ASSERT_EQ(config.partitionedOutputBatchRows, 100000);
  ASSERT_EQ(config.exchangeCompression, "column-adaptive-freq-pfor-min128");
  ASSERT_TRUE(config.exchangeCompressionPipeline);
  ASSERT_EQ(config.exchangeCompressionPipelineThreads, 2);
  ASSERT_EQ(config.exchangeCompressionMinBytes, 268435456);
  ASSERT_DOUBLE_EQ(config.exchangeCompressionSafetyMargin, 1.5);
}
} // namespace facebook::velox::cudf_velox::test
