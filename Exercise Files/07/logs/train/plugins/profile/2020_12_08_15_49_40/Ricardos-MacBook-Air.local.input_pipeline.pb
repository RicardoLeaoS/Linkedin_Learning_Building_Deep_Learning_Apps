	m�����@m�����@!m�����@	lG�OG@lG�OG@!lG�OG@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$m�����@���S��?A�p=
ף�?YZd;�O��?*	     �@2F
Iterator::Modeld;�O���?!�!`�hMU@)=
ףp=�?1�^�7��T@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate1�Zd�?!��۫��!@)J+��?10ߛ�� @:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeath��|?5�?!��b�Vd@)/�$��?1ZT/ߛ@:Preprocessing2U
Iterator::Model::ParallelMapV2�I+��?!ؿp�o��?)�I+��?1ؿp�o��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�� �rh�?!���J�Y�?)�� �rh�?1���J�Y�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip
ףp=
�?!���D��-@){�G�z�?19�7vNK�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;�O��n�?!�(�-��?);�O��n�?1�(�-��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�"��~j�?!�j�="@)����Mbp?1�\Ƒ	�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 46.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s9.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9lG�OG@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���S��?���S��?!���S��?      ��!       "      ��!       *      ��!       2	�p=
ף�?�p=
ף�?!�p=
ף�?:      ��!       B      ��!       J	Zd;�O��?Zd;�O��?!Zd;�O��?R      ��!       Z	Zd;�O��?Zd;�O��?!Zd;�O��?JCPU_ONLYYlG�OG@b 