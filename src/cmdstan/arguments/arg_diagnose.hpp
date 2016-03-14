#ifndef STAN_SERVICES_ARGUMENTS_ARG_DIAGNOSE_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_DIAGNOSE_HPP

#include <cmdstan/arguments/categorical_argument.hpp>
#include <cmdstan/arguments/arg_test.hpp>

namespace stan {
  namespace services {

    class arg_diagnose: public categorical_argument {
    public:
      arg_diagnose() {
        _name = "diagnose";
        _description = "Model diagnostics";

        _subarguments.push_back(new arg_test());
      }
    };

  }  // services
}  // stan

#endif

