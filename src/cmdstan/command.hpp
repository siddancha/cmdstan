#ifndef CMDSTAN_COMMAND_HPP
#define CMDSTAN_COMMAND_HPP

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/additive_combine.hpp>  // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>

#include <stan/version.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/json_parser.hpp>

#include <stan/services/arguments/arg_adapt.hpp>
#include <stan/services/arguments/arg_adapt_delta.hpp>
#include <stan/services/arguments/arg_adapt_engaged.hpp>
#include <stan/services/arguments/arg_adapt_gamma.hpp>
#include <stan/services/arguments/arg_adapt_init_buffer.hpp>
#include <stan/services/arguments/arg_adapt_kappa.hpp>
#include <stan/services/arguments/arg_adapt_t0.hpp>
#include <stan/services/arguments/arg_adapt_term_buffer.hpp>
#include <stan/services/arguments/arg_adapt_window.hpp>
#include <stan/services/arguments/arg_bfgs.hpp>
#include <stan/services/arguments/arg_data.hpp>
#include <stan/services/arguments/arg_data_file.hpp>
#include <stan/services/arguments/arg_dense_e.hpp>
#include <stan/services/arguments/arg_diag_e.hpp>
#include <stan/services/arguments/arg_diagnose.hpp>
#include <stan/services/arguments/arg_diagnostic_file.hpp>
#include <stan/services/arguments/arg_engine.hpp>
#include <stan/services/arguments/arg_fail.hpp>
#include <stan/services/arguments/arg_fixed_param.hpp>
#include <stan/services/arguments/arg_history_size.hpp>
#include <stan/services/arguments/arg_hmc.hpp>
#include <stan/services/arguments/arg_id.hpp>
#include <stan/services/arguments/arg_init.hpp>
#include <stan/services/arguments/arg_init_alpha.hpp>
#include <stan/services/arguments/arg_int_time.hpp>
#include <stan/services/arguments/arg_iter.hpp>
#include <stan/services/arguments/arg_lbfgs.hpp>
#include <stan/services/arguments/arg_max_depth.hpp>
#include <stan/services/arguments/arg_method.hpp>
#include <stan/services/arguments/arg_metric.hpp>
#include <stan/services/arguments/arg_newton.hpp>
#include <stan/services/arguments/arg_num_samples.hpp>
#include <stan/services/arguments/arg_num_warmup.hpp>
#include <stan/services/arguments/arg_nuts.hpp>
#include <stan/services/arguments/arg_optimize.hpp>
#include <stan/services/arguments/arg_optimize_algo.hpp>
#include <stan/services/arguments/arg_output.hpp>
#include <stan/services/arguments/arg_output_file.hpp>
#include <stan/services/arguments/arg_random.hpp>
#include <stan/services/arguments/arg_refresh.hpp>
#include <stan/services/arguments/arg_rwm.hpp>
#include <stan/services/arguments/arg_sample.hpp>
#include <stan/services/arguments/arg_sample_algo.hpp>
#include <stan/services/arguments/arg_save_iterations.hpp>
#include <stan/services/arguments/arg_save_warmup.hpp>
#include <stan/services/arguments/arg_seed.hpp>
#include <stan/services/arguments/arg_static.hpp>
#include <stan/services/arguments/arg_stepsize.hpp>
#include <stan/services/arguments/arg_stepsize_jitter.hpp>
#include <stan/services/arguments/arg_test.hpp>
#include <stan/services/arguments/arg_test_grad_eps.hpp>
#include <stan/services/arguments/arg_test_grad_err.hpp>
#include <stan/services/arguments/arg_test_gradient.hpp>
#include <stan/services/arguments/arg_thin.hpp>
#include <stan/services/arguments/arg_tolerance.hpp>
#include <stan/services/arguments/arg_unit_e.hpp>
#include <stan/services/arguments/argument.hpp>
#include <stan/services/arguments/argument_parser.hpp>
#include <stan/services/arguments/argument_probe.hpp>
#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/list_argument.hpp>
#include <stan/services/arguments/singleton_argument.hpp>
#include <stan/services/arguments/unvalued_argument.hpp>
#include <stan/services/arguments/valued_argument.hpp>
#include <stan/services/diagnose/diagnose.hpp>
#include <stan/services/sample/fixed_param.hpp>
#include <stan/services/sample/hmc_static_unit_e.hpp>
#include <stan/services/sample/hmc_static_unit_e_adapt.hpp>
#include <stan/services/sample/hmc_static_diag_e.hpp>
#include <stan/services/sample/hmc_static_diag_e_adapt.hpp>
#include <stan/services/sample/hmc_static_dense_e.hpp>
#include <stan/services/sample/hmc_static_dense_e_adapt.hpp>
#include <stan/services/sample/hmc_nuts_unit_e.hpp>
#include <stan/services/sample/hmc_nuts_unit_e_adapt.hpp>
#include <stan/services/sample/hmc_nuts_diag_e.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <stan/services/sample/hmc_nuts_dense_e.hpp>
#include <stan/services/sample/hmc_nuts_dense_e_adapt.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/mcmc/hmc/static/adapt_unit_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_diag_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_dense_e_static_hmc.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>

#include <stan/model/util.hpp>

#include <stan/variational/advi.hpp>
#include <stan/services/init/initialize_state.hpp>
#include <stan/services/io/do_print.hpp>
#include <stan/services/io/write_error_msg.hpp>
#include <stan/services/io/write_iteration.hpp>
#include <stan/services/io/write_model.hpp>
#include <stan/services/io/write_stan.hpp>
#include <stan/services/mcmc/sample.hpp>
#include <stan/services/mcmc/warmup.hpp>
#include <stan/services/optimize/bfgs.hpp>
#include <stan/services/optimize/lbfgs.hpp>
#include <stan/services/optimize/newton.hpp>
#include <stan/services/sample/init_adapt.hpp>
#include <stan/services/sample/init_nuts.hpp>
#include <stan/services/sample/init_static_hmc.hpp>
#include <stan/services/sample/init_windowed_adapt.hpp>
#include <stan/services/sample/generate_transitions.hpp>
#include <stan/services/sample/progress.hpp>

#include <stan/interface_callbacks/interrupt/noop.hpp>
#include <stan/interface_callbacks/var_context_factory/dump_factory.hpp>
#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>

#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {
  namespace services {

    class null_fstream : public std::fstream {
    public:
      null_fstream() {}
    };

    template <class Model>
    int command(int argc, const char* argv[]) {
      stan::interface_callbacks::writer::stream_writer info(std::cout);
      stan::interface_callbacks::writer::stream_writer err(std::cout);

      stan::services::argument_parser parser;
      parser.push_valid_arg(new stan::services::arg_id());
      parser.push_valid_arg(new stan::services::arg_data());
      parser.push_valid_arg(new stan::services::arg_init());
      parser.push_valid_arg(new stan::services::arg_random());
      parser.push_valid_arg(new stan::services::arg_output());

      int err_code = parser.parse_args(argc, argv, info, err);
      if (err_code != 0) {
        std::cout << "Failed to parse arguments, terminating Stan" << std::endl;
        return err_code;
      }

      if (parser.help_printed())
        return err_code;

      //////////////////////////////////////////////////
      //                  Input/Output                //
      //////////////////////////////////////////////////

      // Data input
      std::string data_file
        = dynamic_cast<stan::services::string_argument*>
        (parser.arg("data")->arg("file"))->value();

      std::fstream data_stream(data_file.c_str(),
                               std::fstream::in);
      stan::io::dump data_var_context(data_stream);
      data_stream.close();

      // Sample output
      std::string output_file = dynamic_cast<stan::services::string_argument*>(
                                parser.arg("output")->arg("file"))->value();
      std::fstream* output_stream = 0;
      if (output_file != "") {
        output_stream = new std::fstream(output_file.c_str(),
                                         std::fstream::out);
      } else {
        output_stream = new null_fstream();
      }


      // Diagnostic output
      std::string diagnostic_file
        = dynamic_cast<stan::services::string_argument*>
          (parser.arg("output")->arg("diagnostic_file"))->value();

      std::fstream* diagnostic_stream = 0;
      if (diagnostic_file != "") {
        diagnostic_stream = new std::fstream(diagnostic_file.c_str(),
                                             std::fstream::out);
      } else {
        diagnostic_stream = new null_fstream();
      }

      // Refresh rate
      int refresh = dynamic_cast<stan::services::int_argument*>(
                    parser.arg("output")->arg("refresh"))->value();

      // Identification
      unsigned int id = dynamic_cast<stan::services::int_argument*>
        (parser.arg("id"))->value();

      stan::interface_callbacks::writer::stream_writer
        sample_writer(*output_stream, "# "),
        diagnostic_writer(*diagnostic_stream, "# ");

      //////////////////////////////////////////////////
      //            Random number generator           //
      //////////////////////////////////////////////////

      unsigned int random_seed = 0;

      stan::services::u_int_argument* random_arg
        = dynamic_cast<stan::services::u_int_argument*>
        (parser.arg("random")->arg("seed"));

      if (random_arg->is_default()) {
        random_seed
          = (boost::posix_time::microsec_clock::universal_time() -
             boost::posix_time::ptime(boost::posix_time::min_date_time))
          .total_milliseconds();

        random_arg->set_value(random_seed);

      } else {
        random_seed = random_arg->value();
      }

      typedef boost::ecuyer1988 rng_t;  // (2**50 = 1T samples, 1000 chains)
      rng_t base_rng(random_seed);

      // Advance generator to avoid process conflicts
      static boost::uintmax_t DISCARD_STRIDE
        = static_cast<boost::uintmax_t>(1) << 50;
      base_rng.discard(DISCARD_STRIDE * (id - 1));


      //////////////////////////////////////////////////
      //                Initialize Model              //
      //////////////////////////////////////////////////

      Model model(data_var_context, &std::cout);

      Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(model.num_params_r());

      parser.print(info);
      info();

      if (output_stream) {
        io::write_stan(sample_writer);
        io::write_model(sample_writer, model.model_name());
        parser.print(sample_writer);
      }

      if (diagnostic_stream) {
        io::write_stan(diagnostic_writer);
        io::write_model(diagnostic_writer, model.model_name());
        parser.print(diagnostic_writer);
      }

      std::string init = dynamic_cast<stan::services::string_argument*>(
                         parser.arg("init"))->value();

      interface_callbacks::var_context_factory::dump_factory var_context_factory;
      if (!init::initialize_state<interface_callbacks::var_context_factory::dump_factory>
          (init, cont_params, model, base_rng, info,
           var_context_factory))
        return stan::services::error_codes::SOFTWARE;

      //////////////////////////////////////////////////
      //               Model Diagnostics              //
      //////////////////////////////////////////////////

      if (parser.arg("method")->arg("diagnose")) {
        stan::services::list_argument* test = dynamic_cast<stan::services::list_argument*>
                              (parser.arg("method")->arg("diagnose")->arg("test"));

        if (test->value() == "gradient") {
          double epsilon = dynamic_cast<stan::services::real_argument*>
                           (test->arg("gradient")->arg("epsilon"))->value();

          double error = dynamic_cast<stan::services::real_argument*>
                         (test->arg("gradient")->arg("error"))->value();

          return stan::services::diagnose::diagnose(cont_params, model,
                                                    epsilon, error,
                                                    info, sample_writer);

        }
      }


      interface_callbacks::interrupt::noop interrupt;

      //////////////////////////////////////////////////
      //           Optimization Algorithms            //
      //////////////////////////////////////////////////

      if (parser.arg("method")->arg("optimize")) {
        int return_code;
        stan::services::list_argument* algo = dynamic_cast<stan::services::list_argument*>
                              (parser.arg("method")->arg("optimize")->arg("algorithm"));

        int num_iterations = dynamic_cast<stan::services::int_argument*>(
                             parser.arg("method")->arg("optimize")->arg("iter"))->value();

        bool save_iterations
          = dynamic_cast<stan::services::bool_argument*>(parser.arg("method")
                                         ->arg("optimize")
                                         ->arg("save_iterations"))->value();

        if (algo->value() == "newton") {
          return_code = stan::services::optimize::newton(model, base_rng,
                                                         cont_params,
                                                         num_iterations,
                                                         save_iterations,
                                                         interrupt,
                                                         info,
                                                         sample_writer);
        } else if (algo->value() == "bfgs") {
          double init_alpha
            = dynamic_cast<stan::services::real_argument*>
            (algo->arg("bfgs")->arg("init_alpha"))->value();
          double tol_obj
            = dynamic_cast<stan::services::real_argument*>
            (algo->arg("bfgs")->arg("tol_obj"))->value();
          double tol_rel_obj
            = dynamic_cast<stan::services::real_argument*>
            (algo->arg("bfgs")->arg("tol_rel_obj"))->value();
          double tol_grad
            = dynamic_cast<stan::services::real_argument*>
            (algo->arg("bfgs")->arg("tol_grad"))->value();
          double tol_rel_grad
            = dynamic_cast<stan::services::real_argument*>
            (algo->arg("bfgs")->arg("tol_rel_grad"))->value();
          double tol_param
            = dynamic_cast<stan::services::real_argument*>
            (algo->arg("bfgs")->arg("tol_param"))->value();

          return_code = stan::services::optimize::bfgs(model, base_rng,
                                                       cont_params,
                                                       init_alpha,
                                                       tol_obj,
                                                       tol_rel_obj,
                                                       tol_grad,
                                                       tol_rel_grad,
                                                       tol_param,
                                                       num_iterations,
                                                       save_iterations,
                                                       refresh,
                                                       interrupt,
                                                       info, sample_writer);
        } else if (algo->value() == "lbfgs") {
          int history_size
            = dynamic_cast<services::int_argument*>
            (algo->arg("lbfgs")->arg("history_size"))->value();
          double init_alpha
            = dynamic_cast<stan::services::real_argument*>
            (algo->arg("lbfgs")->arg("init_alpha"))->value();
          double tol_obj
            = dynamic_cast<stan::services::real_argument*>
            (algo->arg("lbfgs")->arg("tol_obj"))->value();
          double tol_rel_obj
            = dynamic_cast<stan::services::real_argument*>
            (algo->arg("lbfgs")->arg("tol_rel_obj"))->value();
          double tol_grad
            = dynamic_cast<stan::services::real_argument*>
            (algo->arg("lbfgs")->arg("tol_grad"))->value();
          double tol_rel_grad
            = dynamic_cast<stan::services::real_argument*>
            (algo->arg("lbfgs")->arg("tol_rel_grad"))->value();
          double tol_param
            = dynamic_cast<stan::services::real_argument*>
            (algo->arg("lbfgs")->arg("tol_param"))->value();

          return_code = stan::services::optimize::lbfgs(model, base_rng,
                                                        cont_params,
                                                        history_size,
                                                        init_alpha,
                                                        tol_obj,
                                                        tol_rel_obj,
                                                        tol_grad,
                                                        tol_rel_grad,
                                                        tol_param,
                                                        num_iterations,
                                                        save_iterations,
                                                        refresh,
                                                        interrupt,
                                                        info,
                                                        sample_writer);
        } else {
          return_code = stan::services::error_codes::CONFIG;
        }
        output_stream->close();
        diagnostic_stream->close();
        delete output_stream;
        delete diagnostic_stream;
        return return_code;
      }

      //////////////////////////////////////////////////
      //              Sampling Algorithms             //
      //////////////////////////////////////////////////

      if (parser.arg("method")->arg("sample")) {

        stan::services::list_argument* algo
          = dynamic_cast<stan::services::list_argument*>
          (parser.arg("method")->arg("sample")->arg("algorithm"));

        if (model.num_params_r() == 0 && algo->value() != "fixed_param") {
          info("Must use algorithm=fixed_param for "
               "model that has no parameters.");
          return stan::services::error_codes::USAGE;
        }

        if (algo->value() == "fixed_param") {
          int num_warmup = dynamic_cast<stan::services::int_argument*>(
                            parser.arg("method")->arg("sample")->arg("num_warmup"))->value();

          int num_samples = dynamic_cast<stan::services::int_argument*>(
                            parser.arg("method")->arg("sample")->arg("num_samples"))->value();

          int num_thin = dynamic_cast<stan::services::int_argument*>(
                         parser.arg("method")->arg("sample")->arg("thin"))->value();

          if (num_warmup != 0) {
            info("Warning: warmup will be skipped "
                 "for the fixed parameter sampler!");
          }

          return services::sample::fixed_param(model,
                                               base_rng,
                                               cont_params,
                                               num_samples,
                                               num_thin,
                                               refresh,
                                               interrupt,
                                               sample_writer,
                                               diagnostic_writer,
                                               info);
        } else if (algo->value() == "hmc") {
          stan::services::list_argument* engine
            = dynamic_cast<stan::services::list_argument*>
            (algo->arg("hmc")->arg("engine"));
          
          stan::services::list_argument* metric
            = dynamic_cast<stan::services::list_argument*>
            (algo->arg("hmc")->arg("metric"));

          stan::services::categorical_argument* adapt
            = dynamic_cast<stan::services::categorical_argument*>
            (parser.arg("method")->arg("sample")->arg("adapt"));
          bool adapt_engaged
            = dynamic_cast<stan::services::bool_argument*>(adapt->arg("engaged"))
            ->value();
          
          int num_warmup = dynamic_cast<stan::services::int_argument*>(
                          parser.arg("method")->arg("sample")->arg("num_warmup"))->value();

          int num_samples = dynamic_cast<stan::services::int_argument*>(
                          parser.arg("method")->arg("sample")->arg("num_samples"))->value();

          int num_thin = dynamic_cast<stan::services::int_argument*>(
                       parser.arg("method")->arg("sample")->arg("thin"))->value();

          bool save_warmup = dynamic_cast<stan::services::bool_argument*>(
                       parser.arg("method")->arg("sample")->arg("save_warmup"))->value();

          stan::services::categorical_argument* hmc
            = dynamic_cast<stan::services::categorical_argument*>
            (algo->arg("hmc"));
          
          double stepsize
            = dynamic_cast<stan::services::real_argument*>
            (hmc->arg("stepsize"))->value();
          double stepsize_jitter
            = dynamic_cast<stan::services::real_argument*>
            (hmc->arg("stepsize_jitter"))->value();

          if (engine->value() == "static"
              && metric->value() == "unit_e"
              && adapt_engaged == false) {
            stan::services::categorical_argument* base
              = dynamic_cast<stan::services::categorical_argument*>
              (algo->arg("hmc")->arg("engine")->arg("static"));

            double int_time
              = dynamic_cast<stan::services::real_argument*>(base->arg("int_time"))
              ->value();
            
            return stan::services::sample::hmc_static_unit_e(model,
                                                             base_rng,
                                                             cont_params,
                                                             num_warmup,
                                                             num_samples,
                                                             num_thin,
                                                             save_warmup,
                                                             refresh,
                                                             stepsize,
                                                             stepsize_jitter,
                                                             int_time,
                                                             interrupt,
                                                             sample_writer,
                                                             diagnostic_writer,
                                                             info);
          } else if (engine->value() == "static"
              && metric->value() == "unit_e"
              && adapt_engaged == true) {

            stan::services::categorical_argument* base
              = dynamic_cast<stan::services::categorical_argument*>
              (algo->arg("hmc")->arg("engine")->arg("static"));

            double int_time
              = dynamic_cast<stan::services::real_argument*>(base->arg("int_time"))
              ->value();

            stan::services::categorical_argument* adapt
              = dynamic_cast<stan::services::categorical_argument*>
              (parser.arg("method")->arg("sample")->arg("adapt"));

            double delta
              = dynamic_cast<real_argument*>(adapt->arg("delta"))->value();
            double gamma
              = dynamic_cast<real_argument*>(adapt->arg("gamma"))->value();
            double kappa
              = dynamic_cast<real_argument*>(adapt->arg("kappa"))->value();
            double t0
              = dynamic_cast<real_argument*>(adapt->arg("t0"))->value();
            
            return stan::services::sample::hmc_static_unit_e_adapt(model,
                                                                   base_rng,
                                                                   cont_params,
                                                                   num_warmup,
                                                                   num_samples,
                                                                   num_thin,
                                                                   save_warmup,
                                                                   refresh,
                                                                   stepsize,
                                                                   stepsize_jitter,
                                                                   int_time,
                                                                   delta,
                                                                   gamma,
                                                                   kappa,
                                                                   t0,
                                                                   interrupt,
                                                                   sample_writer,
                                                                   diagnostic_writer,
                                                                   info);            
          } else if (engine->value() == "static"
                     && metric->value() == "diag_e"
                     && adapt_engaged == false) {
            stan::services::categorical_argument* base
              = dynamic_cast<stan::services::categorical_argument*>
              (algo->arg("hmc")->arg("engine")->arg("static"));

            double int_time
              = dynamic_cast<stan::services::real_argument*>(base->arg("int_time"))
              ->value();
            
            return stan::services::sample::hmc_static_diag_e(model,
                                                             base_rng,
                                                             cont_params,
                                                             num_warmup,
                                                             num_samples,
                                                             num_thin,
                                                             save_warmup,
                                                             refresh,
                                                             stepsize,
                                                             stepsize_jitter,
                                                             int_time,
                                                             interrupt,
                                                             sample_writer,
                                                             diagnostic_writer,
                                                             info);
          } else if (engine->value() == "static"
                     && metric->value() == "diag_e"
                     && adapt_engaged == true) {
            stan::services::categorical_argument* base
              = dynamic_cast<stan::services::categorical_argument*>
              (algo->arg("hmc")->arg("engine")->arg("static"));

            double int_time
              = dynamic_cast<stan::services::real_argument*>(base->arg("int_time"))
              ->value();

            stan::services::categorical_argument* adapt
              = dynamic_cast<stan::services::categorical_argument*>
              (parser.arg("method")->arg("sample")->arg("adapt"));

            double delta
              = dynamic_cast<real_argument*>(adapt->arg("delta"))->value();
            double gamma
              = dynamic_cast<real_argument*>(adapt->arg("gamma"))->value();
            double kappa
              = dynamic_cast<real_argument*>(adapt->arg("kappa"))->value();
            double t0
              = dynamic_cast<real_argument*>(adapt->arg("t0"))->value();

            unsigned int init_buffer
              = dynamic_cast<u_int_argument*>(adapt->arg("init_buffer"))->value();
            unsigned int term_buffer
              = dynamic_cast<u_int_argument*>(adapt->arg("term_buffer"))->value();
            unsigned int window
              = dynamic_cast<u_int_argument*>(adapt->arg("window"))->value();

            return stan::services::sample::hmc_static_diag_e_adapt(model,
                                                                   base_rng,
                                                                   cont_params,
                                                                   num_warmup,
                                                                   num_samples,
                                                                   num_thin,
                                                                   save_warmup,
                                                                   refresh,
                                                                   stepsize,
                                                                   stepsize_jitter,
                                                                   int_time,
                                                                   delta,
                                                                   gamma,
                                                                   kappa,
                                                                   t0,
                                                                   init_buffer,
                                                                   term_buffer,
                                                                   window,
                                                                   interrupt,
                                                                   sample_writer,
                                                                   diagnostic_writer,
                                                                   info);
          } else if (engine->value() == "static"
                     && metric->value() == "dense_e"
                     && adapt_engaged == false) {
            stan::services::categorical_argument* base
              = dynamic_cast<stan::services::categorical_argument*>
              (algo->arg("hmc")->arg("engine")->arg("static"));

            double int_time
              = dynamic_cast<stan::services::real_argument*>(base->arg("int_time"))
              ->value();
            
            return stan::services::sample::hmc_static_dense_e(model,
                                                              base_rng,
                                                              cont_params,
                                                              num_warmup,
                                                              num_samples,
                                                              num_thin,
                                                              save_warmup,
                                                              refresh,
                                                              stepsize,
                                                              stepsize_jitter,
                                                              int_time,
                                                              interrupt,
                                                              sample_writer,
                                                              diagnostic_writer,
                                                              info);
          } else if (engine->value() == "static"
                     && metric->value() == "dense_e"
                     && adapt_engaged == true) {
            stan::services::categorical_argument* base
              = dynamic_cast<stan::services::categorical_argument*>
              (algo->arg("hmc")->arg("engine")->arg("static"));

            double int_time
              = dynamic_cast<stan::services::real_argument*>(base->arg("int_time"))
              ->value();

            stan::services::categorical_argument* adapt
              = dynamic_cast<stan::services::categorical_argument*>
              (parser.arg("method")->arg("sample")->arg("adapt"));

            double delta
              = dynamic_cast<real_argument*>(adapt->arg("delta"))->value();
            double gamma
              = dynamic_cast<real_argument*>(adapt->arg("gamma"))->value();
            double kappa
              = dynamic_cast<real_argument*>(adapt->arg("kappa"))->value();
            double t0
              = dynamic_cast<real_argument*>(adapt->arg("t0"))->value();

            unsigned int init_buffer
              = dynamic_cast<u_int_argument*>(adapt->arg("init_buffer"))->value();
            unsigned int term_buffer
              = dynamic_cast<u_int_argument*>(adapt->arg("term_buffer"))->value();
            unsigned int window
              = dynamic_cast<u_int_argument*>(adapt->arg("window"))->value();

            return stan::services::sample::hmc_static_dense_e_adapt(model,
                                                                   base_rng,
                                                                   cont_params,
                                                                   num_warmup,
                                                                   num_samples,
                                                                   num_thin,
                                                                   save_warmup,
                                                                   refresh,
                                                                   stepsize,
                                                                   stepsize_jitter,
                                                                   int_time,
                                                                   delta,
                                                                   gamma,
                                                                   kappa,
                                                                   t0,
                                                                   init_buffer,
                                                                   term_buffer,
                                                                   window,
                                                                   interrupt,
                                                                   sample_writer,
                                                                   diagnostic_writer,
                                                                   info);
          } else if (engine->value() == "nuts"
                     && metric->value() == "unit_e"
                     && adapt_engaged == false) {
            stan::services::categorical_argument* base
              = dynamic_cast<stan::services::categorical_argument*>
              (algo->arg("hmc")->arg("engine")->arg("nuts"));

            int max_depth
              = dynamic_cast<stan::services::int_argument*>(base->arg("max_depth"))
              ->value();
            
            return stan::services::sample::hmc_nuts_unit_e(model,
                                                           base_rng,
                                                           cont_params,
                                                           num_warmup,
                                                           num_samples,
                                                           num_thin,
                                                           save_warmup,
                                                           refresh,
                                                           stepsize,
                                                           stepsize_jitter,
                                                           max_depth,
                                                           interrupt,
                                                           sample_writer,
                                                           diagnostic_writer,
                                                           info);
          } else if (engine->value() == "nuts"
                     && metric->value() == "unit_e"
                     && adapt_engaged == true) {
            stan::services::categorical_argument* base
              = dynamic_cast<stan::services::categorical_argument*>
              (algo->arg("hmc")->arg("engine")->arg("nuts"));

            int max_depth
              = dynamic_cast<stan::services::int_argument*>(base->arg("max_depth"))
              ->value();
            
            stan::services::categorical_argument* adapt
              = dynamic_cast<stan::services::categorical_argument*>
              (parser.arg("method")->arg("sample")->arg("adapt"));

            double delta
              = dynamic_cast<real_argument*>(adapt->arg("delta"))->value();
            double gamma
              = dynamic_cast<real_argument*>(adapt->arg("gamma"))->value();
            double kappa
              = dynamic_cast<real_argument*>(adapt->arg("kappa"))->value();
            double t0
              = dynamic_cast<real_argument*>(adapt->arg("t0"))->value();

            return stan::services::sample::hmc_nuts_unit_e_adapt(model,
                                                                 base_rng,
                                                                 cont_params,
                                                                 num_warmup,
                                                                 num_samples,
                                                                 num_thin,
                                                                 save_warmup,
                                                                 refresh,
                                                                 stepsize,
                                                                 stepsize_jitter,
                                                                 max_depth,
                                                                 delta,
                                                                 gamma,
                                                                 kappa,
                                                                 t0,
                                                                 interrupt,
                                                                 sample_writer,
                                                                 diagnostic_writer,
                                                                 info);
          } else if (engine->value() == "nuts"
                     && metric->value() == "diag_e"
                     && adapt_engaged == false) {
            stan::services::categorical_argument* base
              = dynamic_cast<stan::services::categorical_argument*>
              (algo->arg("hmc")->arg("engine")->arg("nuts"));

            int max_depth
              = dynamic_cast<stan::services::int_argument*>(base->arg("max_depth"))
              ->value();
            
            return stan::services::sample::hmc_nuts_diag_e(model,
                                                           base_rng,
                                                           cont_params,
                                                           num_warmup,
                                                           num_samples,
                                                           num_thin,
                                                           save_warmup,
                                                           refresh,
                                                           stepsize,
                                                           stepsize_jitter,
                                                           max_depth,
                                                           interrupt,
                                                           sample_writer,
                                                           diagnostic_writer,
                                                           info);
          } else if (engine->value() == "nuts"
                     && metric->value() == "diag_e"
                     && adapt_engaged == true) {



            stan::services::categorical_argument* base
              = dynamic_cast<stan::services::categorical_argument*>
              (algo->arg("hmc")->arg("engine")->arg("nuts"));

            int max_depth
              = dynamic_cast<stan::services::int_argument*>(base->arg("max_depth"))
              ->value();
            
            stan::services::categorical_argument* adapt
              = dynamic_cast<stan::services::categorical_argument*>
              (parser.arg("method")->arg("sample")->arg("adapt"));

            double delta
              = dynamic_cast<real_argument*>(adapt->arg("delta"))->value();
            double gamma
              = dynamic_cast<real_argument*>(adapt->arg("gamma"))->value();
            double kappa
              = dynamic_cast<real_argument*>(adapt->arg("kappa"))->value();
            double t0
              = dynamic_cast<real_argument*>(adapt->arg("t0"))->value();

            unsigned int init_buffer
              = dynamic_cast<u_int_argument*>(adapt->arg("init_buffer"))->value();
            unsigned int term_buffer
              = dynamic_cast<u_int_argument*>(adapt->arg("term_buffer"))->value();
            unsigned int window
              = dynamic_cast<u_int_argument*>(adapt->arg("window"))->value();
            
            return stan::services::sample::hmc_nuts_diag_e_adapt(model,
                                                                 base_rng,
                                                                 cont_params,
                                                                 num_warmup,
                                                                 num_samples,
                                                                 num_thin,
                                                                 save_warmup,
                                                                 refresh,
                                                                 stepsize,
                                                                 stepsize_jitter,
                                                                 max_depth,
                                                                 delta,
                                                                 gamma,
                                                                 kappa,
                                                                 t0,
                                                                 init_buffer,
                                                                 term_buffer,
                                                                 window,
                                                                 interrupt,
                                                                 sample_writer,
                                                                 diagnostic_writer,
                                                                 info);
          } else if (engine->value() == "nuts"
                     && metric->value() == "dense_e"
                     && adapt_engaged == false) {
            stan::services::categorical_argument* base
              = dynamic_cast<stan::services::categorical_argument*>
              (algo->arg("hmc")->arg("engine")->arg("nuts"));

            int max_depth
              = dynamic_cast<stan::services::int_argument*>(base->arg("max_depth"))
              ->value();
            
            return stan::services::sample::hmc_nuts_dense_e(model,
                                                           base_rng,
                                                           cont_params,
                                                           num_warmup,
                                                           num_samples,
                                                           num_thin,
                                                           save_warmup,
                                                           refresh,
                                                           stepsize,
                                                           stepsize_jitter,
                                                           max_depth,
                                                           interrupt,
                                                           sample_writer,
                                                           diagnostic_writer,
                                                           info);
          } else if (engine->value() == "nuts"
                     && metric->value() == "dense_e"
                     && adapt_engaged == true) {


            stan::services::categorical_argument* base
              = dynamic_cast<stan::services::categorical_argument*>
              (algo->arg("hmc")->arg("engine")->arg("nuts"));

            int max_depth
              = dynamic_cast<stan::services::int_argument*>(base->arg("max_depth"))
              ->value();
            
            stan::services::categorical_argument* adapt
              = dynamic_cast<stan::services::categorical_argument*>
              (parser.arg("method")->arg("sample")->arg("adapt"));

            double delta
              = dynamic_cast<real_argument*>(adapt->arg("delta"))->value();
            double gamma
              = dynamic_cast<real_argument*>(adapt->arg("gamma"))->value();
            double kappa
              = dynamic_cast<real_argument*>(adapt->arg("kappa"))->value();
            double t0
              = dynamic_cast<real_argument*>(adapt->arg("t0"))->value();

            unsigned int init_buffer
              = dynamic_cast<u_int_argument*>(adapt->arg("init_buffer"))->value();
            unsigned int term_buffer
              = dynamic_cast<u_int_argument*>(adapt->arg("term_buffer"))->value();
            unsigned int window
              = dynamic_cast<u_int_argument*>(adapt->arg("window"))->value();
            
            return stan::services::sample::hmc_nuts_dense_e_adapt(model,
                                                                 base_rng,
                                                                 cont_params,
                                                                 num_warmup,
                                                                 num_samples,
                                                                 num_thin,
                                                                 save_warmup,
                                                                 refresh,
                                                                 stepsize,
                                                                 stepsize_jitter,
                                                                 max_depth,
                                                                 delta,
                                                                 gamma,
                                                                 kappa,
                                                                 t0,
                                                                 init_buffer,
                                                                 term_buffer,
                                                                 window,
                                                                 interrupt,
                                                                 sample_writer,
                                                                 diagnostic_writer,
                                                                 info);
          }
        }

        return stan::services::error_codes::CONFIG;
      }

      //////////////////////////////////////////////////
      //           VARIATIONAL Algorithms             //
      //////////////////////////////////////////////////
      if (parser.arg("method")->arg("variational")) {
        stan::services::list_argument* algo
          = dynamic_cast<stan::services::list_argument*>(parser.arg("method")
            ->arg("variational")->arg("algorithm"));

        int grad_samples = dynamic_cast<stan::services::int_argument*>
          (parser.arg("method")->arg("variational")
                               ->arg("grad_samples"))->value();

        int elbo_samples = dynamic_cast<stan::services::int_argument*>
          (parser.arg("method")->arg("variational")
                               ->arg("elbo_samples"))->value();

        int max_iterations = dynamic_cast<stan::services::int_argument*>
          (parser.arg("method")->arg("variational")
                               ->arg("iter"))->value();

        double tol_rel_obj = dynamic_cast<stan::services::real_argument*>
          (parser.arg("method")->arg("variational")
                               ->arg("tol_rel_obj"))->value();

        double eta = dynamic_cast<stan::services::real_argument*>
          (parser.arg("method")->arg("variational")
                               ->arg("eta"))->value();

        bool adapt_engaged = dynamic_cast<stan::services::bool_argument*>
          (parser.arg("method")->arg("variational")->arg("adapt")
                                                   ->arg("engaged"))->value();

        int adapt_iterations = dynamic_cast<stan::services::int_argument*>
          (parser.arg("method")->arg("variational")->arg("adapt")
                                                   ->arg("iter"))->value();

        int eval_elbo = dynamic_cast<stan::services::int_argument*>
          (parser.arg("method")->arg("variational")
                               ->arg("eval_elbo"))->value();

        int output_samples = dynamic_cast<stan::services::int_argument*>
          (parser.arg("method")->arg("variational")
                               ->arg("output_samples"))->value();

        // Check timing
        clock_t start_check = clock();

        double init_log_prob;
        Eigen::VectorXd init_grad
          = Eigen::VectorXd::Zero(model.num_params_r());

        stan::model::gradient(model, cont_params, init_log_prob,
                              init_grad, &std::cout);

        clock_t end_check = clock();
        double deltaT
          = static_cast<double>(end_check - start_check) / CLOCKS_PER_SEC;

        std::cout << std::endl;
        std::cout << "This is Automatic Differentiation Variational Inference.";
        std::cout << std::endl;

        std::cout << std::endl;
        std::cout << "(EXPERIMENTAL ALGORITHM: expect frequent updates to the"
                  << " procedure.)";
        std::cout << std::endl;

        std::cout << std::endl;
        std::cout << "Gradient evaluation took " << deltaT
                  << " seconds" << std::endl;
        std::cout << "1000 iterations under these settings should take "
                  << 1e3 * grad_samples * deltaT << " seconds." << std::endl;
        std::cout << "Adjust your expectations accordingly!";
        std::cout << std::endl;
        std::cout << std::endl;

        if (algo->value() == "fullrank") {
          std::vector<std::string> names;
          names.push_back("lp__");
          model.constrained_param_names(names, true, true);

          sample_writer(names);

          stan::variational::advi<Model,
                                  stan::variational::normal_fullrank,
                                  rng_t>
            cmd_advi(model,
                     cont_params,
                     base_rng,
                     grad_samples,
                     elbo_samples,
                     eval_elbo,
                     output_samples);
          cmd_advi.run(eta, adapt_engaged, adapt_iterations,
                       tol_rel_obj, max_iterations,
                       info, sample_writer, diagnostic_writer);
        }

        if (algo->value() == "meanfield") {
          std::vector<std::string> names;
          names.push_back("lp__");
          model.constrained_param_names(names, true, true);
          sample_writer(names);

          stan::variational::advi<Model,
                                  stan::variational::normal_meanfield,
                                  rng_t>
            cmd_advi(model,
                     cont_params,
                     base_rng,
                     grad_samples,
                     elbo_samples,
                     eval_elbo,
                     output_samples);
          cmd_advi.run(eta, adapt_engaged, adapt_iterations,
                       tol_rel_obj, max_iterations,
                       info, sample_writer, diagnostic_writer);
        }
      }

      if (output_stream) {
        output_stream->close();
        delete output_stream;
      }

      if (diagnostic_stream) {
        diagnostic_stream->close();
        delete diagnostic_stream;
      }

      return 0;
    }

  }  // namespace services
}  // namespace stan

#endif
