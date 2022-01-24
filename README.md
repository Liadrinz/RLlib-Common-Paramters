⚠请打开目录查找想要的超参数

# 知乎文章

https://zhuanlan.zhihu.com/p/461340889

# 总览

```python
COMMON_CONFIG: TrainerConfigDict = {
    # === Rollout Worker进程的配置 ===
    # 并行采样的rollout worker actors的数量. 设为0会强制在训练器actor中进行rollout.
    "num_workers": 2,
    # 每个worker中以向量化的方式进行策略评估的环境的数量. 可以将推断过程批量化, 
    # 从而解决推断过程的瓶颈
    "num_envs_per_worker": 1,
    # 是否在driver(即用户进程)中创建环境.
    # 当`num_workers` > 0时, driver中不需要环境. 因为driver既不需要做采样, 
    # 也不需要做评估. 采样由远程worker来做(worker_indices > 0), 评估由评估
    # worker来做(详见下方配置).
    "create_env_on_driver": False,
    # 在rollout时将episode划分为长度为`rollout_fragment_length`的片段.
    # rollout worker会收集众多长度为`rollout_fragment_length`的sample batches
    # 并将它们拼接成一个大小为`train_batch_size`的大的batch.
    #
    # 例如, 给定rollout_fragment_length=100和train_batch_size=1000:
    #   1. RLlib将从众多rollout workers中收集10个长度为100的片段.
    #   2. 这些片段将被拼接成大的batch, 并在其上做一轮SGD.
    #
    # 当在每个worker中使用多个环境时, 片段的大小将是原来的`num_envs_per_worker`倍.
    # 因为此时RLlib需要从多个环境中并行地采集每个步骤的信息. 例如, 当
    # num_envs_per_worker > 5时, rollout workers会返回长度为5*100 = 500的片段.
    #
    # 不同的算法在此处有不同的数据流. 例如PPO会进一步将拼接得到的train batch划分为多个
    # minibatches进行多轮的SGD
    "rollout_fragment_length": 200,
    # 每个RolloutWorker构建batch的方式(这里的batch指的是小的sample batches,
    # 它们最终会被拼接为train batch). 以下表述中的"步骤"在配置了不同的
    # `count_steps_by`时表示不同的意思:
    # truncate_episodes: 从RolloutWorker.sample()产生的每个batch将被截断至恰好
    #   包含`rollout_fragment_length`个步骤. 该模式可以保证每个batch尺寸相同,
    #   但是需要计算位于截断边缘的步骤的未来多步的回报, 而非阶段边缘位置只需计算一
    #   步奖励, 从而增加了trajectory中奖励的变化程度.
    # complete_episodes: 每次采样恰好完整地包含一个episode. 数据收集会一直持续到
    #   episode结束或达到所配置的horizon值(可以是硬性的也可以是软性的).
    "batch_mode": "truncate_episodes",

    # === Trainer进程的配置 ===
    # MDP折扣率
    "gamma": 0.99,
    # 默认学习率
    "lr": 0.0001,
    # 训练批的大小(仅在所配数值可用时生效). 应大于等于rollout_fragment_length.
    # Sample batches会被拼接为该大小的batch, 并送入SGD.
    "train_batch_size": 200,
    # 传给策略模型的参数. 完整的模型可用参数详见models/catalog.py
    "model": MODEL_DEFAULTS,
    # 传给策略优化器的参数. 不同的优化器参数不同.
    "optimizer": {},

    # === 环境配置 ===
    # Episode被强制结束的步骤数. 对于Gym环境, 其默认值为
    # `env.spec.max_episode_steps`(如果指定了的话)
    "horizon": None,
    # 是否使用软性horizon, 即到达horizon时计算未来回报、结束episode, 但不重置环境.
    # 这样做可以允许价值评估和RNN状态在以horizon表示的逻辑episodes之间延续.
    # 仅当horizon != inf时生效.
    "soft_horizon": False,
    # 是否在episode的结尾不对'done'置值.
    # 与`soft_horizon`结合起来工作机制如下:
    # - no_done_at_end=False soft_horizon=False:
    #   在episode结束时重置环境, 且置`done=True`
    # - no_done_at_end=True soft_horizon=False:
    #   在episode结束时重置环境, 但不置`done=True`
    # - no_done_at_end=False soft_horizon=True:
    #   在horizon结束时不重置环境, 但置`done=True`(假装episode结束了)
    # - no_done_at_end=True soft_horizon=True:
    #   在horizon结束时不重置环境, 也不置`done=True`
    "no_done_at_end": False,
    # 指定环境. 既可以是在tune中通过
    # `tune.register_env([name], lambda env_ctx: [env object])`
    # 注册的环境, 也可以是RLlib所支持的字符串表示的环境. 对于后者, RLlib会
    # 尝试将字符串解析为openAI gym、PyBullet、ViZDoomGym等环境, 或是一个合
    # 法的环境类的完整路径, 如`ray.rllib.examples.env.random_env.RandomEnv`.
    "env": None,
    # 显式设置环境的观测空间和动作空间. None表示从给定的环境中自动推测
    "observation_space": None,
    "action_space": None,
    # 传给环境创建器的参数字典, 在环境创建器中将构建一个EnvContext对象, 其中包括
    # 一个字典加上num_workers, worker_index, vector_index和remote字段.
    "env_config": {},
    # 如果num_envs_per_worker > 1, 则该配置表示是否在远程进程中而不是同一个worker
    # 中创建这些环境. 这会增加开销, 但是在需要很多时间来进行step或reset的环境中很有用
    # (例如: 星际争霸). 由于开销至关重要, 请慎用此配置.
    "remote_worker_envs": False,
    # 远程worker对环境进行轮询时的等待超时时间.
    # 0表示至少一个环境就绪时继续采样, 是一个合理的默认值.
    # 也可以根据环境的step或reset的时间以及模型推断时间来确定该配置的最优值.
    "remote_env_batch_wait_ms": 0,
    # 一个可调用的对象, 入参为最后一次训练的结果、基础环境和环境上下文,
    # 返回需要指定给该环境的一个新任务. 这里的环境必须是`TaskSettableEnv`
    # 的子类. 使用案例详见`examples/curriculum_learning.py`.
    "env_task_fn": None,
    # 为True时, 将会尝试在本地worker或worker 1(当num_workers > 0时)上渲染环境.
    # 对于向量化的环境, 通常只有第一个子环境会被渲染.
    # 该配置生效的前提时环境需要实现满足如下条件的`render()`方法:
    # a) 能创建窗口并将自身渲染到窗口中(返回True), 或
    # b) 返回一个形状为 [高x宽x3(RGB)] 的uint8类型的numpy数组作为图像
    "render_env": False,
    # 为True时, 会将渲染得到的视频保存到该相对路径下(~/ray_results/...).
    # 你也可以指定一个绝对路径来存储该视频.
    # 为False时将不会记录任何东西.
    # 注意: 该配置代替了弃用的`monitor`配置
    "record_env": False,
    # 在策略后处理时是否对奖励进行裁剪.
    # None (默认值): 仅裁剪Atari环境 (r=sign(r))
    # True: 裁剪所有环境 (r=sign(r)), 从而将只存在-1.0, 1.0和0.0三种奖励
    # False: 从不裁剪
    # [浮点数值value]: 将奖励裁剪至-value至+value之间
    # Tuple[value1, value2]: 将奖励裁剪至value1和value2之间
    "clip_rewards": None,
    # 为True时, RLlib将完全在一个归一化的动作空间中(均值为0.0, 方差较小; 仅影响Box
    # 类型的空间). 但是在动作被送回环境之前会将其复原到原动作空间(反归一化, 反裁剪).
    "normalize_actions": True,
    # 为True时, RLlib将在动作被送回环境前根据环境的边界对动作进行裁剪.
    # TODO: (sven) This option should be obsoleted and always be False.
    # TODO: 该选项已过时, 应永远置为False.
    "clip_actions": False,
    # 默认使用"rllib"还是"deepmind"的预处理器.
    # 如果不需要使用预处理器, 则置为None. 此时, 处理复杂观测的过程可能需要在模型中实现.
    "preprocessor_pref": "deepmind",

    # === 调试配置 ===
    # 为ray.rllib.*中的智能体进程及其worker设置日志级别. 应为DEBUG, INFO, WARN, ERROR
    # 中的一项. DEBUG级别会分阶段地打印内部相关数据流的总结信息 (INFO级别仅在开始时打印
    # 一次). 当使用`rllib train`命令时, 可以通过使用`-v`和`-vv`设置为INFO和DEBUG等级.
    "log_level": "WARN",
    # 会在训练的各个阶段触发的回调函数. 更多用法详见`DefaultCallbacks`类和
    # `examples/custom_metrics_and_callbacks.py`.
    "callbacks": DefaultCallbacks,
    # 当worker崩溃时是否尝试继续训练. 当前健康worker的数量会被记录为
    # `num_healthy_workers`指标.
    "ignore_worker_failures": False,
    # 是否将系统资源使用情况记录到结果中. 打印系统状态需要安装`psutil`,
    # 打印GPU相关指标需要安装`gputil`.
    "log_sys_usage": True,
    # 使用假的采样器(无限速度). 仅供测试.
    "fake_sampler": False,

    # === 深度学习框架配置 ===
    # 指定深度学习框架
    # tf: TensorFlow (静态计算图模式)
    # tf2: TensorFlow 2.x (Eager模式, 如果eager_tracing=True则启用tracing)
    # tfe: TensorFlow eager (Eager模式, 如果eager_tracing=True则启用tracing)
    # torch: PyTorch
    "framework": "tf",
    # 在Eager模式启用tracing. 可以加速2倍左右, 但会提高调试难度, 因为第一次eager
    # 通过后将不会再次解析Python代码. 仅在framework=[tf2|tfe]时有用.
    "eager_tracing": False,
    # tf.function可以重新进行trace(re-trace)的最大次数, 此后将抛出runtime error.
    # 这样做是为了防止在`...eager_traced`策略中对方法进行难以察觉的retraces,
    # 因为这些retraces可能会降低运行速度至1/4且器原因难以被用户察觉.
    # 仅在framework=[tf2|tfe]时需要配置.
    # 如果希望忽略re-trace的次数且从不抛出错误, 则将其设为None
    "eager_max_retraces": 20,

    # === 探索(Exploration)配置 ===
    # 默认的探索行为, 当且仅当`explore`=None传入compute_action(s)时生效.
    # 不需要探索行为时置为False (例如: 用于评估时)
    "explore": True,
    # Exploration对象的配置
    "exploration_config": {
        # 希望使用的Exploration类. 最简单的情况是指定`rllib.utils.exploration`中
        # 的类名字符串. 也可以直接在此指定一个python类, 或是类的完整路径字符串(例如:
        # "ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy").
        "type": "StochasticSampling",
        # 如必要, 在此处添加传给构造函数kwargs的参数
    },
    # === 评估(Evaluation)配置 ===
    # 评估间隔. 每`evaluation_interval`次训练迭代进行一次评估.
    # 评估结果会打印在"evaluation"指标中.
    # 注意: 对于Ape-X指标, 目前只会打印具有最小epsilon的(随机性最小的)workers的结果.
    # None或0表示不进行评估.
    "evaluation_interval": None,
    # 训练时每次评估持续的时间.
    # 时间单位可通过`evalution_duration_unit`来设置, 可以设为`episodes`或`timesteps`,
    # 例如`evaluation_duration = 10`且`evaluation_duration_unit = "episodes"`时,
    # 将运行10个episodes的评估. 如果使用多个评估workers (即evaluation_num_workers > 1),
    # 运行负载将会在它们之间进行分配.
    # 如果`evaluation_duration`为"auto":
    # - 当`evaluation_parallel_to_training=True`时, 将会运行与训练同样多
    #   episodes/timesteps的评估(与训练并行).
    # - 当`evaluation_parallel_to_training=False`时报错.
    "evaluation_duration": 10,
    # 用来计量`evaluation_duration`的单位. 必须为"episodes"(默认值)或"timesteps"
    "evaluation_duration_unit": "episodes",
    # 是否使用线程并行地运行训练过程(Trainer.train())和评估过程. 默认为False.
    # 例如: evalution_interval=2 -> 每隔一次(每两次)训练迭代, Trainer.train()
    # 和Trainer.evaluate()将会并行地运行.
    # 注意: 该配置还是实验性的. 一种可能的缺陷是, 在评估循环开始时的参数同步中可能
    # 存在资源访问冲突(Trainer.evaluate()需要读参数, Trainer.train()需要写参数)
    "evaluation_parallel_to_training": False,
    # 内部标志, 在评估workers中被设为True
    "in_evaluation": False,
    # 评估时对环境的配置.
    # 典型的用法是向评估环境创建器中传入额外的参数,
    # 以及在评估环境中通过计算确定性的动作来禁用探索.
    # IMPORTANT NOTE: 对于策略梯度方法找到的最优策略本身就具有随机性. 此时将"explore"设为
    # False可能会导致评估时使用的并不是最优策略!
    "evaluation_config": {
        # 示例用法: 覆盖env_config, exploration等:
        # "env_config": {...},
        # "explore": False
    },
    # 评估时使用并行workers的数量. 默认情况下为0, 表示评估将会在训练器的进程中运行(仅当
    # evaluation_interval不为None时). 如果增加该值, 则会增加训练器对Ray资源的使用, 但
    # 不会增加rollout workers的负载. 虽然评估过程也需要rollout, 但评估workers与rollout
    # workers是分开创建的, rollout worker特指为训练进行rollout的worker.
    "evaluation_num_workers": 0,
    # 自定义评估方法. 函数原型须满足:
    # (trainer: Trainer, eval_workers: WorkerSet) -> metrics: dict.
    # 默认实现详见Trainer.evaluate()方法.
    # Trainer会在该函数被调用前保证所有评估workers都具有最新的策略状态.
    "custom_eval_function": None,
    # 确保最新可用的评估结果总能记录到每一步(训练)的结果中.
    # 将该项置为True的好处之一是可以保证Tune或其他元数据控制器每一步都能访问到评估结果,
    # 而不是只能在进行了评估的步骤访问到.
    "always_attach_evaluation_results": False,

    # === 高级Rollout设置 ===
    # 使用后台线程进行采样 (轻微off-policy. 除非环境明确需要, 否则通常不可用)
    "sample_async": False,

    # 样本收集器所使用的类, 用于收集和获取环境、模型和采样器数据. 可通过重写
    # SampleCollector基类来实现你自己的数据 收集/缓存/获取 逻辑.
    "sample_collector": SimpleListCollector,

    # 观测滤波器, 对单个观测进行滤波(Element-wise), 必须是"NoFilter"或"MeanStdFilter".
    "observation_filter": "NoFilter",
    # 是否同步远程滤波器的数据
    "synchronize_filters": True,
    # 配置TensorFlow以使其默认以单进程进行操作
    "tf_session_args": {
        # 注意: 以下配置会被`local_tf_session_args`覆盖
        "intra_op_parallelism_threads": 2,
        "inter_op_parallelism_threads": 2,
        "gpu_options": {
            "allow_growth": True,
        },
        "log_device_placement": False,
        "device_count": {
            "CPU": 1
        },
        # 多GPU(num_gpus > 1)必选
        "allow_soft_placement": True,
    },
    # 在本地TensorFlow会话中覆盖这些设置:
    "local_tf_session_args": {
        # 允许更高的默认并行度, 但也不是无限高, 因为过多并发的用户程序可能导致崩溃
        "intra_op_parallelism_threads": 8,
        "inter_op_parallelism_threads": 8,
    },
    # 是否对每个观测进行LZ4压缩
    "compress_observations": False,
    # 等待metric batches的最长时间. 在规定时间内未返回的batches将
    # 在下一次训练迭代时再收集.
    "metrics_episode_collection_timeout_s": 180,
    # 以这么多个episodes的窗口大小对metrics进行平滑化.
    "metrics_num_episodes_for_smoothing": 100,
    # 运行一次`train()`的最小时间间隔:
    # 如果再一次`step_attempt()`后没有达到该时间, 则会继续进行n次`step_attempt()`,
    # 直至达到该时间. 置为0或None表示不规定最小时间.
    "min_time_s_per_reporting": None,
    # 每次调用`train()`时最小的训练/采样时间步.
    # 这个值不影响学习, 只会影响每次迭代的长度.
    # 如果经过一次`step_attempt()`后, 训练/采样的时间步数量没有达到这个值,
    # 则会再进行n次`step_attempt()`直到达到这个值.
    # 置为0或None表示不规定最小时间步.
    "min_train_timesteps_per_reporting": None,
    "min_sample_timesteps_per_reporting": None,

    # 随机种子. 随机种子与worker_index一起确定每个worker的种子.
    # 相同种子的试验会有相同的结果, 使得实验可复现.
    "seed": None,
    # 为训练器进程设置额外的python环境变量,
    # 例如: {"OMP_NUM_THREADS": "16"}
    "extra_python_environs_for_driver": {},
    # 为worker进程设置额外的python环境变量
    "extra_python_environs_for_worker": {},

    # === 资源配置 ===
    # 为训练器进程分配的GPU的数量. 注意, 并非所有算法都需要为训练器分配GPU.
    # 多GPU目前仅对tf-[PPO/IMPALA/DQN/PG]可用.
    # GPU数量可以是分数. (例如: 可以分配0.3块GPU)
    "num_gpus": 0,
    # 使用虚拟GPU, 即用CPU来模拟GPU的功能.
    # 同样可以通过`num_gpus`参数指定不同数量的GPU
    "_fake_gpus": False,
    # 每个worker分配的CPU数量.
    "num_cpus_per_worker": 1,
    # 每个worker分配的GPU数量. 可以是分数.只有当环境本身需要GPU时(例如: GPU密集型
    # 的游戏)或模型推断需要GPU来加速时才需要配置该项.
    "num_gpus_per_worker": 0,
    # 分配给每个worker的用户定义的Ray资源
    "custom_resources_per_worker": {},
    # 分配给trainer的CPU的数量. 注意: 仅当以Tune运行trainer时才生效.
    # 否则trainer将运行在主程序中.
    "num_cpus_for_driver": 1,
    # 资源置放方案组的产生策略. 即tune.PlacementGroupFactory的strategy参数.
    # 一个资源置放方案组(PlacementGroup)定义了哪些设备(资源)应该总是
    # 共同位于同一个节点上. 例如, 具有两个rollout workers的Trainer在以num_gpus=1
    # 的配置运行时会请求一个具有如下形式的资源置放方案组:
    # [{"gpu": 1, "cpu": 1}, {"cpu": 1}, {"cpu": 1}], 其中第一个是driver(trainer)
    # 的资源bundle(由于driver既需要cpu也需要gpu, 这两个资源必须放在一起, 形成一个bundle);
    # 后两个是两个worker的资源bundle.
    # 这些资源bundles可以被"放置"到相同或不同的节点上, 放置的策略取决于`placement_strategy`:
    #  - "PACK": 将bundles打包至需要尽可能少的节点. 例如, 假设集群中有一台机器有1个GPU
    #    和3个CPU, 其他机器都只有一个CPU且没有GPU, 那么"PACK"策略就会将上述PlacementGroup
    #    打包为[{"gpu": 1, "cpu": 3}]放置到这一台机器上. 打包结果就是以下代码的运行结果:
    #    tune.PlacementGroupFactory(
    #        [{"gpu": 1, "cpu": 1}, {"cpu": 1}, {"cpu": 1}], strategy="PACK")
    #  - "SPREAD": 将bundles尽可能均匀地放置到不同的节点上.
    #  - "STRICT_PACK": 将bundles全部打包到一个节点上, 不允许分配到多个节点.
    #  - "STRICT_SPREAD": 将bundles放置到不同的节点上.
    "placement_strategy": "PACK",

    # === 离线数据集配置 ===
    # 指定产生经验的方法:
    #  - "sampler": 通过在线(环境)模拟产生经验. (默认值)
    #  - 指定本地目录或glob表达式: 从该路径读取离线经验 (例如: "/tmp/*.json")
    #  - 指定文件路径/URI列表: 读取这些文件作为离线经验 (例如: ["/tmp/1.json",
    #    "s3://bucket/2.json"])
    #  - 指定一个字典, 键为字符串, 代表一种经验产生方式, 值为采用该方式的概率 (例如:
    #    {"sampler": 0.4, "/tmp/*.json": 0.4, "s3://bucket/expert.json": 0.2})
    #  - 指定一个可调用对象, 以`IOContext`对象作为唯一的参数,
    #    返回ray.rllib.offline.InputReader.
    #  - 指定一个通过tune.registry.regiter_input注册的输入的索引
    "input": "sampler",
    # IOContext可访问的参数, 用于配置用户输入
    "input_config": {},
    # 如果离线数据中的动作已经被归一化 (介于-1.0到1.0之间), 则需要设为True.
    # 例如, 离线数据是由另一个RLlib算法(如PPO或SAC)在"normalize_actions"为True
    # 的配置下运行产生的, 就会发生这种情况.
    "actions_in_input_normalized": False,
    # 指定如何评估当前策略. 仅当读取离线数据时生效(即"input"不为"sampler")时.
    # 可用选项:
    #  - "wis": 使用weighted step-wise importance sampling评估器
    #  - "is": 使用step-wise importance sampling评估器
    #  - "simulation": 在后台运行环境, 但仅用于评估而不用于学习
    "input_evaluation": ["is", "wis"],
    # 是否在离线输入的trajectory片段上运行postprocess_trajectory(). 注意, 在off-policy
    # 算法中, 后处理会使用*当前*策略来完成, 而不会使用*行动*策略.
    "postprocess_inputs": False,
    # 如果为正数, 将使用一个这么大的滑动窗口将输入的batches打乱. 当输入的数据不够随机
    # 时可以使用该配置. 这样做输入会有一定的延迟, 因为需要先将shuffle buffer填满.
    "shuffle_buffer_size": 0,
    # 指定于何处存储经验:
    #  - None: 不存储经验
    #  - "logdir": 存到智能体的日志路径下
    #  - 指定路径/URI(例如: "s3://bucket/")
    #  - 一个返回rllib.offline.OutputWriter的函数
    "output": None,
    # 指定对sample batch中的哪些字段在输出中进行LZ4压缩.
    "output_compress_columns": ["obs", "new_obs"],
    # 单个输出文件的最大大小(如超出则将分一个新的文件).
    "output_max_file_size": 64 * 1024 * 1024,

    # === 多智能体环境配置 ===
    # 多智能体配置
    "multiagent": {
        # 键为策略id, 值为四元组: (策略类, 观测空间, 动作空间, 策略配置).
        # 定义了策略的观测和动作空间和一些额外配置.
        "policies": {},
        # 将最近不常用的策略写入磁盘前, 要在策略表中保留这么多个策略
        # (即保留前`policy_map_capacity`个最近最常用的策略)
        "policy_map_capacity": 100,
        # 将溢出的(最近最不常用的)策略存到`policy_map_cache`.
        # 可以指定一个目录或S3位置. 指定为None则使用默认输出路径.
        "policy_map_cache": None,
        # 将智能体id映射为策略id的函数
        "policy_mapping_fn": None,
        # 指定一个策略列表, 只训练这些策略. 如果为None则训练所有策略.
        "policies_to_train": None,
        # 一个可选函数, 用于增强智能体的局部观测, 使其包含更多的状态.
        # 详见rllib/evaluation/observation_function.py
        "observation_fn": None,
        # 当replay_mode=lockstep时, RLlib会将所有智能体在某个特定时间步的transitions
        # 一起进行回放, 并根据智能体的数量将这些transition作成一个batch. 这样做使得策略
        # 可以在其于该时间步所控制的所有智能体之间执行共享的可微计算(如梯度计算). 换言之,
        # 一些策略在一个时间步控制了多个智能体同时执行动作, lockstep的配置会将这些智能体
        # 的transition同时回放出来, 形成一个batch. 由于这个batch中包含了各个不同的策略
        # 控制的智能体所形成的transitions, 因此将该batch应用于任意一个策略的更新都可以
        # 视为不同策略之间进行了共享.
        # 当replay_mode=independent时, 每个策略独立地进行transitions的回放.
        "replay_mode": "independent",
        # 在构建MultiAgentBatch时, 使用哪个指标作为"batch size", 可选两个值:
        #  - env_steps: 以环境的"step"被调用的次数(不管传入了多少个multi-agent动作,
        #    也不管上一步返回了多少个multi-agent观测)作为batch size.
        #  - agent_steps: 以所有智能体经历的steps之和作为batch size.
        "count_steps_by": "env_steps",
    },

    # === 日志记录器配置 ===
    # 针对特定的logger定义Logger对象中使用的configuration
    # 可以使用嵌套的字典来替代默认值None
    "logger_config": None,
}
```

# Rollout Worker进程的配置

示例:

```python
{
    # ...
    "num_workers": 2,
    "num_envs_per_worker": 1,
    "create_env_on_driver": False,
    "rollout_fragment_length": 200,
    "batch_mode": "truncate_episodes",
    # ...
}
```
## num_workers


```python
{
    # 并行采样的rollout worker actors的数量. 设为0会强制在训练器actor中进行rollout.
    "num_workers": 2,
}
```

## num_envs_per_worker


```python
{
    # 每个worker中以向量化的方式进行策略评估的环境的数量. 可以将推断过程批量化, 
    # 从而解决推断过程的瓶颈
    "num_envs_per_worker": 1,
}
```

## create_env_on_driver


```python
{
    # 是否在driver(即用户进程)中创建环境.
    # 当`num_workers` > 0时, driver中不需要环境. 因为driver既不需要做采样, 
    # 也不需要做评估. 采样由远程worker来做(worker_indices > 0), 评估由评估
    # worker来做(详见下方配置).
    "create_env_on_driver": False,
}
```

## rollout_fragment_length


```python
{
    # 在rollout时将episode划分为长度为`rollout_fragment_length`的片段.
    # rollout worker会收集众多长度为`rollout_fragment_length`的sample batches
    # 并将它们拼接成一个大小为`train_batch_size`的大的batch.
    #
    # 例如, 给定rollout_fragment_length=100和train_batch_size=1000:
    #   1. RLlib将从众多rollout workers中收集10个长度为100的片段.
    #   2. 这些片段将被拼接成大的batch, 并在其上做一轮SGD.
    #
    # 当在每个worker中使用多个环境时, 片段的大小将是原来的`num_envs_per_worker`倍.
    # 因为此时RLlib需要从多个环境中并行地采集每个步骤的信息. 例如, 当
    # num_envs_per_worker > 5时, rollout workers会返回长度为5*100 = 500的片段.
    #
    # 不同的算法在此处有不同的数据流. 例如PPO会进一步将拼接得到的train batch划分为多个
    # minibatches进行多轮的SGD
    "rollout_fragment_length": 200,
}
```

## batch_mode


```python
{
    # 每个RolloutWorker构建batch的方式(这里的batch指的是小的sample batches,
    # 它们最终会被拼接为train batch). 以下表述中的"步骤"在配置了不同的
    # `count_steps_by`时表示不同的意思:
    # truncate_episodes: 从RolloutWorker.sample()产生的每个batch将被截断至恰好
    #   包含`rollout_fragment_length`个步骤. 该模式可以保证每个batch尺寸相同,
    #   但是需要计算位于截断边缘的步骤的未来多步的回报, 而非阶段边缘位置只需计算一
    #   步奖励, 从而增加了trajectory中奖励的变化程度.
    # complete_episodes: 每次采样恰好完整地包含一个episode. 数据收集会一直持续到
    #   episode结束或达到所配置的horizon值(可以是硬性的也可以是软性的).
    "batch_mode": "truncate_episodes",

}
```



# Trainer进程的配置

示例:

```python
{
    # ...
    "gamma": 0.99,
    "lr": 0.0001,
    "train_batch_size": 200,
    "model": MODEL_DEFAULTS,
    "optimizer": {},
    # ...
}
```
## gamma


```python
{
    # MDP折扣率
    "gamma": 0.99,
}
```

## lr


```python
{
    # 默认学习率
    "lr": 0.0001,
}
```

## train_batch_size


```python
{
    # 训练批的大小(仅在所配数值可用时生效). 应大于等于rollout_fragment_length.
    # Sample batches会被拼接为该大小的batch, 并送入SGD.
    "train_batch_size": 200,
}
```

## model


```python
{
    # 传给策略模型的参数. 完整的模型可用参数详见models/catalog.py
    "model": MODEL_DEFAULTS,
}
```

## optimizer


```python
{
    # 传给策略优化器的参数. 不同的优化器参数不同.
    "optimizer": {},

}
```



# 环境配置

示例:

```python
{
    # ...
    "horizon": None,
    "soft_horizon": False,
    "no_done_at_end": False,
    "env": None,
    "observation_space": None,
    "action_space": None,
    "env_config": {},
    "remote_worker_envs": False,
    "remote_env_batch_wait_ms": 0,
    "env_task_fn": None,
    "render_env": False,
    "record_env": False,
    "clip_rewards": None,
    "normalize_actions": True,
    "clip_actions": False,
    "preprocessor_pref": "deepmind",
    # ...
}
```
## horizon


```python
{
    # Episode被强制结束的步骤数. 对于Gym环境, 其默认值为
    # `env.spec.max_episode_steps`(如果指定了的话)
    "horizon": None,
}
```

## soft_horizon


```python
{
    # 是否使用软性horizon, 即到达horizon时计算未来回报、结束episode, 但不重置环境.
    # 这样做可以允许价值评估和RNN状态在以horizon表示的逻辑episodes之间延续.
    # 仅当horizon != inf时生效.
    "soft_horizon": False,
}
```

## no_done_at_end


```python
{
    # 是否在episode的结尾不对'done'置值.
    # 与`soft_horizon`结合起来工作机制如下:
    # - no_done_at_end=False soft_horizon=False:
    #   在episode结束时重置环境, 且置`done=True`
    # - no_done_at_end=True soft_horizon=False:
    #   在episode结束时重置环境, 但不置`done=True`
    # - no_done_at_end=False soft_horizon=True:
    #   在horizon结束时不重置环境, 但置`done=True`(假装episode结束了)
    # - no_done_at_end=True soft_horizon=True:
    #   在horizon结束时不重置环境, 也不置`done=True`
    "no_done_at_end": False,
}
```

## env


```python
{
    # 指定环境. 既可以是在tune中通过
    # `tune.register_env([name], lambda env_ctx: [env object])`
    # 注册的环境, 也可以是RLlib所支持的字符串表示的环境. 对于后者, RLlib会
    # 尝试将字符串解析为openAI gym、PyBullet、ViZDoomGym等环境, 或是一个合
    # 法的环境类的完整路径, 如`ray.rllib.examples.env.random_env.RandomEnv`.
    "env": None,
}
```

## observation_space


```python
{
    # 显式设置环境的观测空间和动作空间. None表示从给定的环境中自动推测
    "observation_space": None,
    "action_space": None,
}
```

## env_config


```python
{
    # 传给环境创建器的参数字典, 在环境创建器中将构建一个EnvContext对象, 其中包括
    # 一个字典加上num_workers, worker_index, vector_index和remote字段.
    "env_config": {},
}
```

## remote_worker_envs


```python
{
    # 如果num_envs_per_worker > 1, 则该配置表示是否在远程进程中而不是同一个worker
    # 中创建这些环境. 这会增加开销, 但是在需要很多时间来进行step或reset的环境中很有用
    # (例如: 星际争霸). 由于开销至关重要, 请慎用此配置.
    "remote_worker_envs": False,
}
```

## remote_env_batch_wait_ms


```python
{
    # 远程worker对环境进行轮询时的等待超时时间.
    # 0表示至少一个环境就绪时继续采样, 是一个合理的默认值.
    # 也可以根据环境的step或reset的时间以及模型推断时间来确定该配置的最优值.
    "remote_env_batch_wait_ms": 0,
}
```

## env_task_fn


```python
{
    # 一个可调用的对象, 入参为最后一次训练的结果、基础环境和环境上下文,
    # 返回需要指定给该环境的一个新任务. 这里的环境必须是`TaskSettableEnv`
    # 的子类. 使用案例详见`examples/curriculum_learning.py`.
    "env_task_fn": None,
}
```

## render_env


```python
{
    # 为True时, 将会尝试在本地worker或worker 1(当num_workers > 0时)上渲染环境.
    # 对于向量化的环境, 通常只有第一个子环境会被渲染.
    # 该配置生效的前提时环境需要实现满足如下条件的`render()`方法:
    # a) 能创建窗口并将自身渲染到窗口中(返回True), 或
    # b) 返回一个形状为 [高x宽x3(RGB)] 的uint8类型的numpy数组作为图像
    "render_env": False,
}
```

## record_env


```python
{
    # 为True时, 会将渲染得到的视频保存到该相对路径下(~/ray_results/...).
    # 你也可以指定一个绝对路径来存储该视频.
    # 为False时将不会记录任何东西.
    # 注意: 该配置代替了弃用的`monitor`配置
    "record_env": False,
}
```

## clip_rewards


```python
{
    # 在策略后处理时是否对奖励进行裁剪.
    # None (默认值): 仅裁剪Atari环境 (r=sign(r))
    # True: 裁剪所有环境 (r=sign(r)), 从而将只存在-1.0, 1.0和0.0三种奖励
    # False: 从不裁剪
    # [浮点数值value]: 将奖励裁剪至-value至+value之间
    # Tuple[value1, value2]: 将奖励裁剪至value1和value2之间
    "clip_rewards": None,
}
```

## normalize_actions


```python
{
    # 为True时, RLlib将完全在一个归一化的动作空间中(均值为0.0, 方差较小; 仅影响Box
    # 类型的空间). 但是在动作被送回环境之前会将其复原到原动作空间(反归一化, 反裁剪).
    "normalize_actions": True,
}
```

## clip_actions


```python
{
    # 为True时, RLlib将在动作被送回环境前根据环境的边界对动作进行裁剪.
    # TODO: (sven) This option should be obsoleted and always be False.
    # TODO: 该选项已过时, 应永远置为False.
    "clip_actions": False,
}
```

## preprocessor_pref


```python
{
    # 默认使用"rllib"还是"deepmind"的预处理器.
    # 如果不需要使用预处理器, 则置为None. 此时, 处理复杂观测的过程可能需要在模型中实现.
    "preprocessor_pref": "deepmind",

}
```



# 调试配置

示例:

```python
{
    # ...
    "log_level": "WARN",
    "callbacks": DefaultCallbacks,
    "ignore_worker_failures": False,
    "log_sys_usage": True,
    "fake_sampler": False,
    # ...
}
```
## log_level


```python
{
    # 为ray.rllib.*中的智能体进程及其worker设置日志级别. 应为DEBUG, INFO, WARN, ERROR
    # 中的一项. DEBUG级别会分阶段地打印内部相关数据流的总结信息 (INFO级别仅在开始时打印
    # 一次). 当使用`rllib train`命令时, 可以通过使用`-v`和`-vv`设置为INFO和DEBUG等级.
    "log_level": "WARN",
}
```

## callbacks


```python
{
    # 会在训练的各个阶段触发的回调函数. 更多用法详见`DefaultCallbacks`类和
    # `examples/custom_metrics_and_callbacks.py`.
    "callbacks": DefaultCallbacks,
}
```

## ignore_worker_failures


```python
{
    # 当worker崩溃时是否尝试继续训练. 当前健康worker的数量会被记录为
    # `num_healthy_workers`指标.
    "ignore_worker_failures": False,
}
```

## log_sys_usage


```python
{
    # 是否将系统资源使用情况记录到结果中. 打印系统状态需要安装`psutil`,
    # 打印GPU相关指标需要安装`gputil`.
    "log_sys_usage": True,
}
```

## fake_sampler


```python
{
    # 使用假的采样器(无限速度). 仅供测试.
    "fake_sampler": False,

}
```



# 深度学习框架配置

示例:

```python
{
    # ...
    "framework": "tf",
    "eager_tracing": False,
    "eager_max_retraces": 20,
    # ...
}
```
## framework


```python
{
    # 指定深度学习框架
    # tf: TensorFlow (静态计算图模式)
    # tf2: TensorFlow 2.x (Eager模式, 如果eager_tracing=True则启用tracing)
    # tfe: TensorFlow eager (Eager模式, 如果eager_tracing=True则启用tracing)
    # torch: PyTorch
    "framework": "tf",
}
```

## eager_tracing


```python
{
    # 在Eager模式启用tracing. 可以加速2倍左右, 但会提高调试难度, 因为第一次eager
    # 通过后将不会再次解析Python代码. 仅在framework=[tf2|tfe]时有用.
    "eager_tracing": False,
}
```

## eager_max_retraces


```python
{
    # tf.function可以重新进行trace(re-trace)的最大次数, 此后将抛出runtime error.
    # 这样做是为了防止在`...eager_traced`策略中对方法进行难以察觉的retraces,
    # 因为这些retraces可能会降低运行速度至1/4且器原因难以被用户察觉.
    # 仅在framework=[tf2|tfe]时需要配置.
    # 如果希望忽略re-trace的次数且从不抛出错误, 则将其设为None
    "eager_max_retraces": 20,

}
```



# 探索(Exploration)配置

示例:

```python
{
    # ...
    "explore": True,
    "exploration_config": {
        "type": "StochasticSampling",
    },    # ...
}
```
## explore


```python
{
    # 默认的探索行为, 当且仅当`explore`=None传入compute_action(s)时生效.
    # 不需要探索行为时置为False (例如: 用于评估时)
    "explore": True,
}
```

## exploration_config


```python
{
    # Exploration对象的配置
    "exploration_config": {
        # 希望使用的Exploration类. 最简单的情况是指定`rllib.utils.exploration`中
        # 的类名字符串. 也可以直接在此指定一个python类, 或是类的完整路径字符串(例如:
        # "ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy").
        "type": "StochasticSampling",
        # 如必要, 在此处添加传给构造函数kwargs的参数
    },
}
```



# 评估(Evaluation)配置

示例:

```python
{
    # ...
    "evaluation_interval": None,
    "evaluation_duration": 10,
    "evaluation_duration_unit": "episodes",
    "evaluation_parallel_to_training": False,
    "in_evaluation": False,
    "evaluation_config": {
    },
    "evaluation_num_workers": 0,
    "custom_eval_function": None,
    "always_attach_evaluation_results": False,
    # ...
}
```
## evaluation_interval


```python
{
    # 评估间隔. 每`evaluation_interval`次训练迭代进行一次评估.
    # 评估结果会打印在"evaluation"指标中.
    # 注意: 对于Ape-X指标, 目前只会打印具有最小epsilon的(随机性最小的)workers的结果.
    # None或0表示不进行评估.
    "evaluation_interval": None,
}
```

## evaluation_duration


```python
{
    # 训练时每次评估持续的时间.
    # 时间单位可通过`evalution_duration_unit`来设置, 可以设为`episodes`或`timesteps`,
    # 例如`evaluation_duration = 10`且`evaluation_duration_unit = "episodes"`时,
    # 将运行10个episodes的评估. 如果使用多个评估workers (即evaluation_num_workers > 1),
    # 运行负载将会在它们之间进行分配.
    # 如果`evaluation_duration`为"auto":
    # - 当`evaluation_parallel_to_training=True`时, 将会运行与训练同样多
    #   episodes/timesteps的评估(与训练并行).
    # - 当`evaluation_parallel_to_training=False`时报错.
    "evaluation_duration": 10,
}
```

## evaluation_duration_unit


```python
{
    # 用来计量`evaluation_duration`的单位. 必须为"episodes"(默认值)或"timesteps"
    "evaluation_duration_unit": "episodes",
}
```

## evaluation_parallel_to_training


```python
{
    # 是否使用线程并行地运行训练过程(Trainer.train())和评估过程. 默认为False.
    # 例如: evalution_interval=2 -> 每隔一次(每两次)训练迭代, Trainer.train()
    # 和Trainer.evaluate()将会并行地运行.
    # 注意: 该配置还是实验性的. 一种可能的缺陷是, 在评估循环开始时的参数同步中可能
    # 存在资源访问冲突(Trainer.evaluate()需要读参数, Trainer.train()需要写参数)
    "evaluation_parallel_to_training": False,
}
```

## in_evaluation


```python
{
    # 内部标志, 在评估workers中被设为True
    "in_evaluation": False,
}
```

## evaluation_config


```python
{
    # 评估时对环境的配置.
    # 典型的用法是向评估环境创建器中传入额外的参数,
    # 以及在评估环境中通过计算确定性的动作来禁用探索.
    # IMPORTANT NOTE: 对于策略梯度方法找到的最优策略本身就具有随机性. 此时将"explore"设为
    # False可能会导致评估时使用的并不是最优策略!
    "evaluation_config": {
        # 示例用法: 覆盖env_config, exploration等:
        # "env_config": {...},
        # "explore": False
    },
}
```

## evaluation_num_workers


```python
{
    # 评估时使用并行workers的数量. 默认情况下为0, 表示评估将会在训练器的进程中运行(仅当
    # evaluation_interval不为None时). 如果增加该值, 则会增加训练器对Ray资源的使用, 但
    # 不会增加rollout workers的负载. 虽然评估过程也需要rollout, 但评估workers与rollout
    # workers是分开创建的, rollout worker特指为训练进行rollout的worker.
    "evaluation_num_workers": 0,
}
```

## custom_eval_function


```python
{
    # 自定义评估方法. 函数原型须满足:
    # (trainer: Trainer, eval_workers: WorkerSet) -> metrics: dict.
    # 默认实现详见Trainer.evaluate()方法.
    # Trainer会在该函数被调用前保证所有评估workers都具有最新的策略状态.
    "custom_eval_function": None,
}
```

## always_attach_evaluation_results


```python
{
    # 确保最新可用的评估结果总能记录到每一步(训练)的结果中.
    # 将该项置为True的好处之一是可以保证Tune或其他元数据控制器每一步都能访问到评估结果,
    # 而不是只能在进行了评估的步骤访问到.
    "always_attach_evaluation_results": False,

}
```



# 高级Rollout设置

示例:

```python
{
    # ...
    "sample_async": False,

    "sample_collector": SimpleListCollector,

    "observation_filter": "NoFilter",
    "synchronize_filters": True,
    "tf_session_args": {
        "intra_op_parallelism_threads": 2,
        "inter_op_parallelism_threads": 2,
        "gpu_options": {
            "allow_growth": True,
        },
        "log_device_placement": False,
        "device_count": {
            "CPU": 1
        },
        "allow_soft_placement": True,
    },
    "local_tf_session_args": {
        "intra_op_parallelism_threads": 8,
        "inter_op_parallelism_threads": 8,
    },
    "compress_observations": False,
    "metrics_episode_collection_timeout_s": 180,
    "metrics_num_episodes_for_smoothing": 100,
    "min_time_s_per_reporting": None,
    "min_train_timesteps_per_reporting": None,
    "min_sample_timesteps_per_reporting": None,

    "seed": None,
    "extra_python_environs_for_driver": {},
    "extra_python_environs_for_worker": {},
    # ...
}
```
## sample_async


```python
{
    # 使用后台线程进行采样 (轻微off-policy. 除非环境明确需要, 否则通常不可用)
    "sample_async": False,

}
```

## sample_collector


```python
{
    # 样本收集器所使用的类, 用于收集和获取环境、模型和采样器数据. 可通过重写
    # SampleCollector基类来实现你自己的数据 收集/缓存/获取 逻辑.
    "sample_collector": SimpleListCollector,

}
```

## observation_filter


```python
{
    # 观测滤波器, 对单个观测进行滤波(Element-wise), 必须是"NoFilter"或"MeanStdFilter".
    "observation_filter": "NoFilter",
}
```

## synchronize_filters


```python
{
    # 是否同步远程滤波器的数据
    "synchronize_filters": True,
}
```

## tf_session_args


```python
{
    # 配置TensorFlow以使其默认以单进程进行操作
    "tf_session_args": {
        # 注意: 以下配置会被`local_tf_session_args`覆盖
        "intra_op_parallelism_threads": 2,
        "inter_op_parallelism_threads": 2,
        "gpu_options": {
            "allow_growth": True,
        },
        "log_device_placement": False,
        "device_count": {
            "CPU": 1
        },
        # 多GPU(num_gpus > 1)必选
        "allow_soft_placement": True,
    },
}
```

## local_tf_session_args


```python
{
    # 在本地TensorFlow会话中覆盖这些设置:
    "local_tf_session_args": {
        # 允许更高的默认并行度, 但也不是无限高, 因为过多并发的用户程序可能导致崩溃
        "intra_op_parallelism_threads": 8,
        "inter_op_parallelism_threads": 8,
    },
}
```

## compress_observations


```python
{
    # 是否对每个观测进行LZ4压缩
    "compress_observations": False,
}
```

## metrics_episode_collection_timeout_s


```python
{
    # 等待metric batches的最长时间. 在规定时间内未返回的batches将
    # 在下一次训练迭代时再收集.
    "metrics_episode_collection_timeout_s": 180,
}
```

## metrics_num_episodes_for_smoothing


```python
{
    # 以这么多个episodes的窗口大小对metrics进行平滑化.
    "metrics_num_episodes_for_smoothing": 100,
}
```

## min_time_s_per_reporting


```python
{
    # 运行一次`train()`的最小时间间隔:
    # 如果再一次`step_attempt()`后没有达到该时间, 则会继续进行n次`step_attempt()`,
    # 直至达到该时间. 置为0或None表示不规定最小时间.
    "min_time_s_per_reporting": None,
}
```

## min_train_timesteps_per_reporting


```python
{
    # 每次调用`train()`时最小的训练/采样时间步.
    # 这个值不影响学习, 只会影响每次迭代的长度.
    # 如果经过一次`step_attempt()`后, 训练/采样的时间步数量没有达到这个值,
    # 则会再进行n次`step_attempt()`直到达到这个值.
    # 置为0或None表示不规定最小时间步.
    "min_train_timesteps_per_reporting": None,
    "min_sample_timesteps_per_reporting": None,

}
```

## seed


```python
{
    # 随机种子. 随机种子与worker_index一起确定每个worker的种子.
    # 相同种子的试验会有相同的结果, 使得实验可复现.
    "seed": None,
}
```

## extra_python_environs_for_driver


```python
{
    # 为训练器进程设置额外的python环境变量,
    # 例如: {"OMP_NUM_THREADS": "16"}
    "extra_python_environs_for_driver": {},
}
```

## extra_python_environs_for_worker


```python
{
    # 为worker进程设置额外的python环境变量
    "extra_python_environs_for_worker": {},

}
```



# 资源配置

示例:

```python
{
    # ...
    "num_gpus": 0,
    "_fake_gpus": False,
    "num_cpus_per_worker": 1,
    "num_gpus_per_worker": 0,
    "custom_resources_per_worker": {},
    "num_cpus_for_driver": 1,
    "placement_strategy": "PACK",
    # ...
}
```
## num_gpus


```python
{
    # 为训练器进程分配的GPU的数量. 注意, 并非所有算法都需要为训练器分配GPU.
    # 多GPU目前仅对tf-[PPO/IMPALA/DQN/PG]可用.
    # GPU数量可以是分数. (例如: 可以分配0.3块GPU)
    "num_gpus": 0,
}
```

## _fake_gpus


```python
{
    # 使用虚拟GPU, 即用CPU来模拟GPU的功能.
    # 同样可以通过`num_gpus`参数指定不同数量的GPU
    "_fake_gpus": False,
}
```

## num_cpus_per_worker


```python
{
    # 每个worker分配的CPU数量.
    "num_cpus_per_worker": 1,
}
```

## num_gpus_per_worker


```python
{
    # 每个worker分配的GPU数量. 可以是分数.只有当环境本身需要GPU时(例如: GPU密集型
    # 的游戏)或模型推断需要GPU来加速时才需要配置该项.
    "num_gpus_per_worker": 0,
}
```

## custom_resources_per_worker


```python
{
    # 分配给每个worker的用户定义的Ray资源
    "custom_resources_per_worker": {},
}
```

## num_cpus_for_driver


```python
{
    # 分配给trainer的CPU的数量. 注意: 仅当以Tune运行trainer时才生效.
    # 否则trainer将运行在主程序中.
    "num_cpus_for_driver": 1,
}
```

## placement_strategy


```python
{
    # 资源置放方案组的产生策略. 即tune.PlacementGroupFactory的strategy参数.
    # 一个资源置放方案组(PlacementGroup)定义了哪些设备(资源)应该总是
    # 共同位于同一个节点上. 例如, 具有两个rollout workers的Trainer在以num_gpus=1
    # 的配置运行时会请求一个具有如下形式的资源置放方案组:
    # [{"gpu": 1, "cpu": 1}, {"cpu": 1}, {"cpu": 1}], 其中第一个是driver(trainer)
    # 的资源bundle(由于driver既需要cpu也需要gpu, 这两个资源必须放在一起, 形成一个bundle);
    # 后两个是两个worker的资源bundle.
    # 这些资源bundles可以被"放置"到相同或不同的节点上, 放置的策略取决于`placement_strategy`:
    #  - "PACK": 将bundles打包至需要尽可能少的节点. 例如, 假设集群中有一台机器有1个GPU
    #    和3个CPU, 其他机器都只有一个CPU且没有GPU, 那么"PACK"策略就会将上述PlacementGroup
    #    打包为[{"gpu": 1, "cpu": 3}]放置到这一台机器上. 打包结果就是以下代码的运行结果:
    #    tune.PlacementGroupFactory(
    #        [{"gpu": 1, "cpu": 1}, {"cpu": 1}, {"cpu": 1}], strategy="PACK")
    #  - "SPREAD": 将bundles尽可能均匀地放置到不同的节点上.
    #  - "STRICT_PACK": 将bundles全部打包到一个节点上, 不允许分配到多个节点.
    #  - "STRICT_SPREAD": 将bundles放置到不同的节点上.
    "placement_strategy": "PACK",

}
```



# 离线数据集配置

示例:

```python
{
    # ...
    "input": "sampler",
    "input_config": {},
    "actions_in_input_normalized": False,
    "input_evaluation": ["is", "wis"],
    "postprocess_inputs": False,
    "shuffle_buffer_size": 0,
    "output": None,
    "output_compress_columns": ["obs", "new_obs"],
    "output_max_file_size": 64 * 1024 * 1024,
    # ...
}
```
## input


```python
{
    # 指定产生经验的方法:
    #  - "sampler": 通过在线(环境)模拟产生经验. (默认值)
    #  - 指定本地目录或glob表达式: 从该路径读取离线经验 (例如: "/tmp/*.json")
    #  - 指定文件路径/URI列表: 读取这些文件作为离线经验 (例如: ["/tmp/1.json",
    #    "s3://bucket/2.json"])
    #  - 指定一个字典, 键为字符串, 代表一种经验产生方式, 值为采用该方式的概率 (例如:
    #    {"sampler": 0.4, "/tmp/*.json": 0.4, "s3://bucket/expert.json": 0.2})
    #  - 指定一个可调用对象, 以`IOContext`对象作为唯一的参数,
    #    返回ray.rllib.offline.InputReader.
    #  - 指定一个通过tune.registry.regiter_input注册的输入的索引
    "input": "sampler",
}
```

## input_config


```python
{
    # IOContext可访问的参数, 用于配置用户输入
    "input_config": {},
}
```

## actions_in_input_normalized


```python
{
    # 如果离线数据中的动作已经被归一化 (介于-1.0到1.0之间), 则需要设为True.
    # 例如, 离线数据是由另一个RLlib算法(如PPO或SAC)在"normalize_actions"为True
    # 的配置下运行产生的, 就会发生这种情况.
    "actions_in_input_normalized": False,
}
```

## input_evaluation


```python
{
    # 指定如何评估当前策略. 仅当读取离线数据时生效(即"input"不为"sampler")时.
    # 可用选项:
    #  - "wis": 使用weighted step-wise importance sampling评估器
    #  - "is": 使用step-wise importance sampling评估器
    #  - "simulation": 在后台运行环境, 但仅用于评估而不用于学习
    "input_evaluation": ["is", "wis"],
}
```

## postprocess_inputs


```python
{
    # 是否在离线输入的trajectory片段上运行postprocess_trajectory(). 注意, 在off-policy
    # 算法中, 后处理会使用*当前*策略来完成, 而不会使用*行动*策略.
    "postprocess_inputs": False,
}
```

## shuffle_buffer_size


```python
{
    # 如果为正数, 将使用一个这么大的滑动窗口将输入的batches打乱. 当输入的数据不够随机
    # 时可以使用该配置. 这样做输入会有一定的延迟, 因为需要先将shuffle buffer填满.
    "shuffle_buffer_size": 0,
}
```

## output


```python
{
    # 指定于何处存储经验:
    #  - None: 不存储经验
    #  - "logdir": 存到智能体的日志路径下
    #  - 指定路径/URI(例如: "s3://bucket/")
    #  - 一个返回rllib.offline.OutputWriter的函数
    "output": None,
}
```

## output_compress_columns


```python
{
    # 指定对sample batch中的哪些字段在输出中进行LZ4压缩.
    "output_compress_columns": ["obs", "new_obs"],
}
```

## output_max_file_size


```python
{
    # 单个输出文件的最大大小(如超出则将分一个新的文件).
    "output_max_file_size": 64 * 1024 * 1024,

}
```



# 多智能体环境配置

示例:

```python
{
    # ...
    "multiagent": {
        "policies": {},
        "policy_map_capacity": 100,
        "policy_map_cache": None,
        "policy_mapping_fn": None,
        "policies_to_train": None,
        "observation_fn": None,
        "replay_mode": "independent",
        "count_steps_by": "env_steps",
    },
    # ...
}
```
## multiagent


```python
{
    # 多智能体配置
    "multiagent": {
        # 键为策略id, 值为四元组: (策略类, 观测空间, 动作空间, 策略配置).
        # 定义了策略的观测和动作空间和一些额外配置.
        "policies": {},
        # 将最近不常用的策略写入磁盘前, 要在策略表中保留这么多个策略
        # (即保留前`policy_map_capacity`个最近最常用的策略)
        "policy_map_capacity": 100,
        # 将溢出的(最近最不常用的)策略存到`policy_map_cache`.
        # 可以指定一个目录或S3位置. 指定为None则使用默认输出路径.
        "policy_map_cache": None,
        # 将智能体id映射为策略id的函数
        "policy_mapping_fn": None,
        # 指定一个策略列表, 只训练这些策略. 如果为None则训练所有策略.
        "policies_to_train": None,
        # 一个可选函数, 用于增强智能体的局部观测, 使其包含更多的状态.
        # 详见rllib/evaluation/observation_function.py
        "observation_fn": None,
        # 当replay_mode=lockstep时, RLlib会将所有智能体在某个特定时间步的transitions
        # 一起进行回放, 并根据智能体的数量将这些transition作成一个batch. 这样做使得策略
        # 可以在其于该时间步所控制的所有智能体之间执行共享的可微计算(如梯度计算). 换言之,
        # 一些策略在一个时间步控制了多个智能体同时执行动作, lockstep的配置会将这些智能体
        # 的transition同时回放出来, 形成一个batch. 由于这个batch中包含了各个不同的策略
        # 控制的智能体所形成的transitions, 因此将该batch应用于任意一个策略的更新都可以
        # 视为不同策略之间进行了共享.
        # 当replay_mode=independent时, 每个策略独立地进行transitions的回放.
        "replay_mode": "independent",
        # 在构建MultiAgentBatch时, 使用哪个指标作为"batch size", 可选两个值:
        #  - env_steps: 以环境的"step"被调用的次数(不管传入了多少个multi-agent动作,
        #    也不管上一步返回了多少个multi-agent观测)作为batch size.
        #  - agent_steps: 以所有智能体经历的steps之和作为batch size.
        "count_steps_by": "env_steps",
    },

}
```



# 日志记录器配置

示例:

```python
{
    # ...
    "logger_config": None,    # ...
}
```
## logger_config


```python
{
    # 针对特定的logger定义Logger对象中使用的configuration
    # 可以使用嵌套的字典来替代默认值None
    "logger_config": None,
}
```



