// Copyright 2019-2025 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#if !__has_include(<boost/json/src.hpp>)
#error BOOST >= 1.75 is required for Boost.JSON
#endif

#include <ygm/comm.hpp>
#include <ygm/container/detail/base_iteration.hpp>
#include <ygm/io/line_parser.hpp>
#include <ygm/utility/boost_json.hpp>

namespace ygm::io {

/**
 * @brief Erase given keys from a JSON object
 *
 * @param obj JSON object to delete key from
 * @param keys Keys to erase
 * @return Number of keys erased
 */
inline std::size_t json_erase(boost::json::object            &obj,
                              const std::vector<std::string> &keys) {
  std::size_t num_erased = 0;
  for (const auto &key : keys) {
    num_erased += obj.erase(key);
  }
  return num_erased;
}

/**
 * @brief Erase all keys from a JSON object except those provided
 *
 * @param obj JSON object to update
 * @param include_keys Keys to leave in JSON object
 * @return Number of keys filtered from JSON object
 */
inline std::size_t json_filter(boost::json::object            &obj,
                               const std::vector<std::string> &include_keys) {
  std::set<std::string>    include_keys_set{include_keys.begin(),
                                         include_keys.end()};
  std::vector<std::string> keys_to_erase;
  for (auto itr = obj.begin(), end = obj.end(); itr != end; ++itr) {
    if (include_keys_set.count(itr->key().data()) == 0) {
      keys_to_erase.emplace_back(itr->key().data());
    }
  }
  return json_erase(obj, keys_to_erase);
}

/**
 * @brief Parser for handling collections of newline-delimited JSON files in
 * parallel.
 */
class ndjson_parser : public ygm::container::detail::base_iteration_value<
                          ndjson_parser, std::tuple<boost::json::object>> {
 public:
  using for_all_args = std::tuple<boost::json::object>;
  template <typename... Args>
  ndjson_parser(Args &&...args) : m_lp(std::forward<Args>(args)...) {}

  class iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type        = boost::json::object;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const boost::json::object *;
    using reference         = const boost::json::object &;

    iterator() = default;  // sentinel/end iterator

    reference operator*() const { return m_impl->m_current_line; }
    pointer   operator->() const { return &m_impl->m_current_line; }

    iterator &operator++() {
      m_impl->advance();
      return *this;
    }

    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const iterator &a, const iterator &b) {
      bool a_end = !a.m_impl || !a.m_impl->m_lp_iter.valid();
      bool b_end = !b.m_impl || !b.m_impl->m_lp_iter.valid();
      if (a_end || b_end) {
        return a_end == b_end;
      }
      return a.m_impl->m_lp_iter == b.m_impl->m_lp_iter;
    }

    friend bool operator!=(const iterator &a, const iterator &b) {
      return !(a == b);
    }

   private:
    friend class ndjson_parser;

    struct impl {
      ygm::io::line_parser::iterator m_lp_iter;
      boost::json::object            m_current_line;
      bool                           m_valid_line;
      const ndjson_parser           *p_parser{nullptr};

      impl() : m_current_line() {};

      // Advances to the next line from the underlying line_parser and parses it
      // as a JSON line
      void advance() {
        do {
          ++m_lp_iter;
          try {
            m_current_line = boost::json::parse(*m_lp_iter).as_object();
            m_valid_line   = true;
          } catch (...) {
            m_valid_line = false;
            // Count invalid line if line parser says line is valid but JSON
            // parsing fails
            if (m_lp_iter.valid()) {
              p_parser->m_num_invalid_records++;
            }
          }
        } while (not m_valid_line and m_lp_iter.valid());
      }
    };

    explicit iterator(std::shared_ptr<impl> impl) : m_impl(std::move(impl)) {}

    std::shared_ptr<impl> m_impl;
  };

  using const_iterator = iterator;

  /**
   * @brief Executes a user function for every CSV record in a set of files.
   *
   * @tparam Function
   * @param fn User function to execute
   */
  template <typename Function>
  void for_all(Function fn) {
    for (const auto &jsonline : *this) {
      fn(jsonline);
    }
  }

  /*
   * @brief Access to underlying communicator
   *
   * @return YGM communicator used by parser
   */
  ygm::comm &comm() { return m_lp.comm(); }

  /*
   * @brief `comm()` function for `const` parsers that returns a `const
   * ygm::comm`
   *
   * @return YGM communicator used by parser
   */
  const ygm::comm &comm() const { return m_lp.comm(); }

  /**
   * @brief Returns an iterator to the first line of CSV assigned to this rank
   */
  iterator begin() const {
    m_num_invalid_records = 0;  // Reset to avoid over-reporting invalid lines
                                // when reading multiple times

    auto impl       = std::make_shared<iterator::impl>();
    impl->m_lp_iter = m_lp.begin();
    impl->p_parser  = this;

    try {
      impl->m_current_line = boost::json::parse(*(impl->m_lp_iter)).as_object();
      impl->m_valid_line   = true;
    } catch (...) {
      impl->m_valid_line = false;
      if (impl->m_lp_iter.valid()) {
        ++m_num_invalid_records;
      }
    }
    return iterator(impl);
  }

  /**
   * @brief Returns a past-the-end sentinel iterator
   */
  iterator end() const { return iterator(); }

  /*
   * @brief Get a count of the number of invalid JSON lines encountered during
   * parsing
   *
   * @return Number of invalid JSON lines
   */
  size_t num_invalid_records() {
    return ygm::sum(m_num_invalid_records, m_lp.comm());
  }

 private:
  line_parser m_lp;

  mutable size_t m_num_invalid_records{0};
};

}  // namespace ygm::io
