from automatic_prompt_engineer import template, data
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from instruction_coupled_kernel import *
import json
import torch
import random
from OpenSourceLLM import OpenSourceLLMAPI

tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

def PromptOptimizer(args):

    task, model_hf_dir = args.task, args.model_hf_dir
    vec_dim, tokens_num = args.vec_dim, args.tokens_num
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    init_prompt = ['\n']
    system_template = "[full_DEMO]\n\nThe instruction for reflection was?"
    system_template = template.InitQATemplate(system_template)
    fin_template = template.DemosTemplate(demos_template)


    model_invocation = OpenSourceLLMAPI(init_prompt=init_prompt, vec_dim=vec_dim, tokens_num=tokens_num, model_hf_dir=model_hf_dir)

    # 随机生成初始10条soft prompt作为贝叶斯优化初始点
    X = SobolEngine(dimension=vec_dim, scramble=True, seed=0).draw(10)

    X_return = []
    #对上述贝叶斯优化的初始点进行评估得到reward
    for x in X:
        # 随机抽取反思样例池中的例子
        with open("./reflection_suggestions_pool/"+args.task+"_reflection_example.json", "r", encoding="utf-8") as f:
            content = json.load(f)

        sample_data = random.sample(content, 2)
        input = []
        output = []
        for data in sample_data:
            input.append(data["input"])
            output.append(data["output"])
        reference_data = (input, output)
        print(reference_data)
        demos = fin_template.fill(reference_data)
        llm_input = [system_template.fill(demos)]
        X_return.append(model_invocation.eval(x, llm_input, 0, args))

    Y = [X[0] for X in X_return]
    Y_scores = [X[1].squeeze() for X in X_return]
    X = X.to(**tkwargs)
    Y = torch.FloatTensor(Y).unsqueeze(-1).to(**tkwargs)
    Y_scores = torch.FloatTensor(np.array(Y_scores)).to(**tkwargs)

    X_train = X
    y_train = (Y - Y.mean(dim=-2)) / (Y.std(dim=-2))

    # 定义高斯核函数
    matern_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=X_train.shape[-1],
        lengthscale_prior=GammaPrior(3.0, 6.0),
    )

    matern_kernel_instruction = MaternKernel(
        nu=2.5,
        ard_num_dims=Y_scores.shape[-1],
        lengthscale_prior=GammaPrior(3.0, 6.0),
    )

    covar_module = ScaleKernel(
        base_kernel=CombinedStringKernel(base_latent_kernel=matern_kernel, instruction_kernel=matern_kernel_instruction,
                                         latent_train=X_train.double(), instruction_train=Y_scores))

    # 定义高斯过程回归模型
    gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module)
    # 定义边缘对数似然
    gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    for i in range(4):
        # 根据最大似然原理去拟合代理模型
        fit_gpytorch_model(gp_mll)
        #定义EI采集函数
        EI = ExpectedImprovement(gp_model, best_f=y_train.max().item())
        print('EI:',EI)

        starting_idxs = torch.argsort(-1 * y_train)[:10]
        starting_points = X_train[starting_idxs]

        best_points = []
        best_vals = []
        #探索新的数据点
        for starting_point_for_cma in starting_points:
            if (torch.max(starting_point_for_cma) > 1 or torch.min(starting_point_for_cma) < -1):
                continue
            newp, newv = cma_es_concat(starting_point_for_cma, EI, tkwargs)
            best_points.append(newp)
            best_vals.append(newv)
        #对新探索到的点进行评估得到reward
        for idx in np.argsort(-1 * np.array(best_vals)):
            X_next_point = torch.from_numpy(best_points[idx]).float().unsqueeze(0)

            #随机抽取反思样例池中的例子
            with open("./reflection_suggestions_pool/"+args.task + "_reflection_example.json", "r", encoding="utf-8") as f:
                content = json.load(f)

            sample_data = random.sample(content, 2)
            input = []
            output = []
            for data in sample_data:
                input.append(data["input"])
                output.append(data["output"])
            reference_data = (input, output)

            demos = fin_template.fill(reference_data)
            llm_input = [system_template.fill(demos)]
            X_next_points_return = [model_invocation.eval(X_next_point, llm_input, i+1, args)]
            Y_next_point = [X[0] for X in X_next_points_return]

            Y_scores_next_points = [X[1].squeeze() for X in X_next_points_return]

            X_next_point = X_next_point.to(**tkwargs)
            Y_next_point = torch.FloatTensor(Y_next_point).unsqueeze(-1).to(**tkwargs)
            Y_scores_next_points = torch.FloatTensor(np.array(Y_scores_next_points)).to(**tkwargs)

            X = torch.cat([X, X_next_point])
            Y = torch.cat([Y, Y_next_point])
            Y_scores = torch.cat([Y_scores, Y_scores_next_points])

        #根据现有的数据点更新核函数、高斯过程回归模型等
        X_train = X.clone()
        y_train = (Y - Y.mean(dim=-2)) / (Y.std(dim=-2))

        matern_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=X_train.shape[-1],
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )

        matern_kernel_instruction = MaternKernel(
            nu=2.5,
            ard_num_dims=Y_scores.shape[-1],
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )
        covar_module = ScaleKernel(base_kernel=CombinedStringKernel(base_latent_kernel=matern_kernel,
                                                                    instruction_kernel=matern_kernel_instruction,
                                                                    latent_train=X_train.double(),
                                                                    instruction_train=Y_scores))
        gp_model = SingleTaskGP(X_train, y_train, covar_module=covar_module)
        gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)


    meta_prompt = model_invocation.best_prompt()
    print("Best prompt is:",meta_prompt)
    return meta_prompt[0], model_invocation.success_fail, model_invocation.success_id